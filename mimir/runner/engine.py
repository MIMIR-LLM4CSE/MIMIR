"""The headless run engine: ``run_benchmark`` drives the agent over an adapter's tasks.

Per task: set up its workspace, start the servers there, run a fresh agent with every
tool approved (trusted workloads only), hand the answer to the adapter's scorer — or, for
a ``BatchAdapter``, keep what it collects and grade every run together at the end. A
failing task is recorded, not fatal; a task missing its toolchain is skipped.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import tempfile
import time
import traceback
from typing import Any, Callable

from mimir.client.extensions import all_servers
from mimir.client.agent_core import MimirAgent
from mimir.client.core.human import Refusing, Unattended
from mimir.client.config import prompt_variants
from mimir.client.core.ports import AgentHost
from mimir.client.guardrails.workflow import is_incomplete_answer
from mimir.client.llm.base import CALL_USAGE_KEYS, call_purpose

from .types import BatchAdapter, BenchmarkAdapter, BenchTask, CheckContext, RunResult


class _CountingBackend:
    """The backend every model call of a run goes through, each call's usage recorded.

    The loop's own calls (its steps) are told from side calls such as a compaction by
    ``call_purpose``. Resolved at its first use, inside the run: a backend that cannot be
    reached fails the task, not the benchmark.
    """

    def __init__(self, get_backend: Any) -> None:
        self._get_backend = get_backend
        self._backend: Any = None
        self.calls = 0
        self.side_calls: dict[str, int] = {}
        self.usage: dict[str, int | None] = dict.fromkeys(CALL_USAGE_KEYS)
        self.prompt_curve: list[int] = []

    def _resolved(self) -> Any:
        if self._backend is None:
            self._backend = self._get_backend()
        return self._backend

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        msg = self._resolved().chat(*args, **kwargs)
        purpose = call_purpose()
        if purpose == "loop":
            self.calls += 1
        else:
            self.side_calls[purpose] = self.side_calls.get(purpose, 0) + 1
        if isinstance(msg, dict):
            for key in CALL_USAGE_KEYS:
                reported = msg.get(key)
                if isinstance(reported, int):
                    self.usage[key] = (self.usage[key] or 0) + reported
            if purpose == "loop" and isinstance(msg.get("prompt_tokens"), int):
                self.prompt_curve.append(msg["prompt_tokens"])
        return msg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolved(), name)


def _install_approver(agent: MimirAgent, approval: str) -> Refusing | None:
    """Lend the agent who answers its approvals: nobody approving all (``auto_all``), or
    nobody refusing all (``refuse``), returned so its record can be read."""
    if approval == "auto_all":
        agent.attach(AgentHost(human=Unattended("benchmark-auto-approve")))
        return None
    if approval == "refuse":
        refusing = Refusing()
        agent.attach(AgentHost(human=refusing))
        return refusing
    raise ValueError(f"unknown approval {approval!r}: auto_all or refuse")


def _hold_context(agent: MimirAgent, task: BenchTask) -> None:
    """Hold the agent to the context *task* asks for."""
    if task.context_mode is not None:
        agent.apply_setting("context_mode", task.context_mode)
    if task.max_context_tokens:
        agent.max_context_tokens = int(task.max_context_tokens)
    agent.intra_query_compaction = task.compaction


async def _run_within(agent: MimirAgent, run: Any, limit: float | None) -> tuple[str, bool]:
    """The answer of *run*, and whether *limit* stopped it first — as Stop does: the cancel
    flag ends a stream at its next chunk, then the turn is cancelled."""
    future = asyncio.ensure_future(run)
    done, _ = await asyncio.wait({future}, timeout=limit)
    if future in done:
        return future.result(), False
    agent.turn.cancel_flag.set()
    future.cancel()
    with contextlib.suppress(BaseException):
        await future
    return "", True


async def run_one(
    task: BenchTask,
    adapter: BenchmarkAdapter | BatchAdapter,
    model: str,
    get_backend_override: Any | None = None,
    enforcement: str | None = None,
    keep_workspace: bool = False,
) -> RunResult:
    """Run *task* through a fresh agent and score it with *adapter*; the workspace goes
    once graded unless *keep_workspace*."""
    # Skip before spending an agent run when a required toolchain is absent.
    missing = missing_requirements(task)
    if missing:
        return RunResult(task.id, False, f"skipped: missing {', '.join(missing)}", skipped=True)

    workspace = tempfile.mkdtemp(prefix=f"bench_{task.id}_")
    if task.setup is not None:
        task.setup(workspace)
    events: list[dict] = []
    orig_cwd = os.getcwd()
    os.chdir(workspace)  # MCP file/search servers root at the cwd they're spawned in

    # The backend every call takes — the live one, or a scripted one (model-free CI mode) —
    # counted on the way through: the loop's (streaming) and the compaction's (agent_core).
    from unittest.mock import patch
    import mimir.client.agent_core as agent_core_module
    import mimir.client.query_engine.streaming as streaming_module
    counter = _CountingBackend(get_backend_override or streaming_module.get_backend)
    backend_patch = contextlib.ExitStack()
    for module in (streaming_module, agent_core_module):
        backend_patch.enter_context(patch.object(module, "get_backend", lambda: counter))
    started = time.monotonic()
    answered: float | None = None
    refusing: Refusing | None = None

    agent = MimirAgent(model=model)
    if enforcement is not None:
        # Override the model-profile default so a whole run can be graded at one
        # nudge level (strict/light/off) regardless of which model is served.
        agent.apply_setting("enforcement", enforcement)
    try:
        with backend_patch:
            refusing = _install_approver(agent, task.approval)
            _hold_context(agent, task)
            registry = all_servers()
            server_names = task.servers if task.servers is not None else tuple(registry)
            for name in server_names:
                if name not in registry:
                    raise KeyError(f"unknown server '{name}' for task '{task.id}'")
            await agent.connect_servers({n: registry[n] for n in server_names})
            agent.seed_classification_from_caps()

            answer, timed_out = await _run_within(agent, agent.run(
                task.query,
                max_steps=task.max_steps,
                mode=task.mode,
                event_callback=events.append,
            ), task.timeout_secs)
            answered = time.monotonic()
            return await graded_result(
                adapter, task, answer, CheckContext(workspace=workspace, agent=agent),
                lambda: _measures(events, counter, started, answered, refusing),
                timed_out=timed_out, incomplete=is_incomplete_answer(answer))
    except Exception:
        return RunResult(task.id, False, "run crashed", error=traceback.format_exc(),
                         **_measures(events, counter, started, answered, refusing))
    finally:
        with contextlib.suppress(Exception):
            await agent.cleanup()
        os.chdir(orig_cwd)
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def missing_requirements(task: BenchTask) -> list[str]:
    """The executables and Python modules *task* needs that this environment lacks."""
    missing = [exe for exe in task.requires if shutil.which(exe) is None]
    return missing + [m for m in task.modules if importlib.util.find_spec(m) is None]


async def graded_result(adapter: BenchmarkAdapter | BatchAdapter, task: BenchTask,
                        answer: str, ctx: CheckContext, measures: Callable[[], dict[str, Any]],
                        **fields: Any) -> RunResult:
    """*task*'s run, scored by *adapter* — or collected, pending, for a ``BatchAdapter`` —
    with *measures* taken once the grade is in, so they count it."""
    if isinstance(adapter, BatchAdapter):
        artifact = await adapter.collect(task, answer, ctx)
        return RunResult(task.id, False, "awaiting the batch grade", pending=True,
                         artifact=artifact, **fields, **measures())
    scored = await adapter.score(task, answer, ctx)
    # A scorer that finds a prerequisite it provides itself missing says so with a third
    # element: the task is skipped, not failed.
    skipped = scored[2] if len(scored) > 2 else False
    return RunResult(task.id, scored[0], scored[1], skipped=skipped, **fields, **measures())


def _measures(events: list[dict], counter: _CountingBackend, started: float,
              answered: float | None = None, refusing: Refusing | None = None) -> dict[str, Any]:
    """What a run cost and what it was told, read off its events, its backend and whoever
    answered its approvals."""
    nudges: dict[str, int] = {}
    context: dict[str, int] = {}
    for event in events:
        if event.get("type") == "nudge_injected":
            category = str(event.get("category") or "?")
            nudges[category] = nudges.get(category, 0) + 1
        elif event.get("type") == "status" and isinstance(event.get("context"), dict):
            kind = str(event["context"].get("kind") or "?")
            context[kind] = context.get(kind, 0) + 1
    now = time.monotonic()
    return {
        "tool_calls": sum(1 for e in events if e.get("type") == "tool_call"),
        "events": len(events),
        "model_calls": counter.calls,
        "side_calls": dict(counter.side_calls),
        **counter.usage,
        "prompt_curve": list(counter.prompt_curve),
        "peak_prompt_tokens": max(counter.prompt_curve, default=None),
        "context_events": context,
        "refused": list(refusing.refused) if refusing is not None else [],
        "nudges": nudges,
        "elapsed_secs": round(now - started, 2),
        "answer_secs": round((answered if answered is not None else now) - started, 2),
        "variant": prompt_variants.active_name(),
    }


def use_backend(name: str) -> None:
    """Make *name* (``vllm``, ``anthropic``…) the backend the next agents call."""
    os.environ["LLM_BACKEND"] = name
    from mimir.client.llm.factory import clear_backend_cache
    clear_backend_cache()


async def grade_pending(adapter: BenchmarkAdapter | BatchAdapter, results: list[RunResult],
                        label: str) -> None:
    """Settle the pending runs of *results* with one ``grade`` call named *label*."""
    pending = [r for r in results if r.pending]
    if not pending or not isinstance(adapter, BatchAdapter):
        return
    try:
        verdicts = await adapter.grade(pending, label)
    except Exception:
        verdicts, error = {}, traceback.format_exc()
    else:
        error = ""
    for result in pending:
        verdict = verdicts.get(result.task_id)
        if verdict is None:
            result.detail = "the grade crashed" if error else "the grader gave no verdict"
            result.error = result.error or error
        else:
            result.passed, result.detail = verdict[0], verdict[1]
            result.skipped = verdict[2] if len(verdict) > 2 else False
        result.pending = False


async def run_benchmark(
    adapter: BenchmarkAdapter | BatchAdapter,
    *,
    model: str,
    backend: str = "vllm",
    report_path: str | None = None,
    limit: int | None = None,
    get_backend_override: Any | None = None,
    enforcement: str | None = None,
    keep_workspaces: bool = False,
) -> dict:
    """Run *adapter*'s tasks; return a summary dict, written to *report_path* if given.

    *get_backend_override* injects a backend (CI); *enforcement* fixes the nudge level for
    every task; *keep_workspaces* leaves each task's workspace on disk. The pass rate
    excludes skipped tasks.
    """
    if get_backend_override is None:
        use_backend(backend)

    tasks = adapter.load_tasks(limit)
    started = time.time()
    results: list[RunResult] = []
    for task in tasks:
        results.append(
            await run_one(task, adapter, model, get_backend_override, enforcement,
                          keep_workspace=keep_workspaces)
        )
    await grade_pending(adapter, results, f"{adapter.name}-{time.strftime('%Y%m%d-%H%M%S')}")

    passed = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results if r.skipped)
    scored = len(results) - skipped
    summary = {
        "benchmark": adapter.name,
        "model": model,
        "backend": "scripted" if get_backend_override is not None else backend,
        "enforcement": enforcement or "model-default",
        "total": len(results),
        "passed": passed,
        "skipped": skipped,
        "pass_rate": round(passed / scored, 4) if scored else 0.0,
        "elapsed_secs": round(time.time() - started, 2),
        "results": [r.as_dict() for r in results],
    }
    if report_path:
        with open(report_path, "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary
