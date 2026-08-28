"""Headless run engine — drives the agent over a list of tasks, adapter-agnostic.

This is the agent's "batch mode": given an adapter (a list of tasks + a scorer),
``run_benchmark`` runs the agent over every task and returns a JSON-serialisable
summary. It does **no scoring of its own** — scoring is the adapter/benchmark's job.

For each task (``run_one``): materialise its workspace (``task.setup``), point the
MCP file/search servers at that workspace (they read ``os.getcwd()`` at spawn — see
``integration/server_manager.connect_server``), drive the agent's non-interactive
path (``allow_continue_prompt`` left False → fixed ``max_steps``, no human prompt),
then hand the answer to the adapter's scorer.

Two things make a run unattended:
- A **fresh ``MimirAgent`` per task** so no state (``_carry_context``,
  ``_tool_cache``, history) leaks between tasks.
- An **auto-approve hook** that replaces the interactive approval gate. Batch
  mode already auto-approves write/edit tools, but ``non_batch_tools`` (code
  execution / shell) would otherwise block on ``input()``; the hook approves
  every tool. WARNING: this runs every tool without confirmation — only point it
  at trusted benchmark workloads in a sandbox.

A single task failing (timeout, crashed server, scorer error) records a failed
result instead of aborting the whole suite; a task whose ``requires`` toolchain is
absent is recorded *skipped* without spending an agent run.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

from mimir.client.extensions import all_servers
from mimir.client.agent_core import MimirAgent

from .types import BenchmarkAdapter, BenchTask, CheckContext


@dataclass
class RunResult:
    task_id: str
    passed: bool
    detail: str
    skipped: bool = False     # toolchain/prereq absent → not run, not counted in pass-rate
    tool_calls: int = 0       # tool_call events observed
    events: int = 0           # total structured events emitted (proxy for run length)
    error: str = ""           # populated when the run itself crashed

    def as_dict(self) -> dict:
        return asdict(self)


def _install_auto_approve(agent: MimirAgent) -> None:
    """Make the agent approve every tool without prompting (unattended mode)."""
    agent.allow_continue_prompt = False  # stop at max_steps; no human continue prompt
    agent._request_tool_approval = (
        lambda name, arguments, max_attempts=3: (True, "benchmark-auto-approve")
    )
    agent._request_path_approval = lambda abspath, tool_name, arguments=None: (True, False)


async def run_one(
    task: BenchTask,
    adapter: BenchmarkAdapter,
    model: str,
    get_backend_override: Any | None = None,
    enforcement: str | None = None,
) -> RunResult:
    """Run *task* through a fresh agent and score it with *adapter*."""
    # Skip before spending an agent run when a required toolchain is absent.
    missing = [exe for exe in task.requires if shutil.which(exe) is None]
    if missing:
        return RunResult(
            task.id, False, f"skipped: missing {', '.join(missing)}", skipped=True,
        )

    workspace = tempfile.mkdtemp(prefix=f"bench_{task.id}_")
    if task.setup is not None:
        task.setup(workspace)
    events: list[dict] = []
    orig_cwd = os.getcwd()
    os.chdir(workspace)  # MCP file/search servers root at the cwd they're spawned in

    # Optionally swap the live backend for a scripted one (model-free CI mode).
    backend_patch = contextlib.nullcontext()
    if get_backend_override is not None:
        from unittest.mock import patch
        import mimir.client.query_engine.streaming as streaming_module
        backend_patch = patch.object(
            streaming_module, "get_backend", get_backend_override
        )

    agent = MimirAgent(model=model)
    if enforcement is not None:
        # Override the model-profile default so a whole run can be graded at one
        # nudge level (strict/light/off) regardless of which model is served.
        agent.set_enforcement(enforcement)
    try:
        with backend_patch:
            _install_auto_approve(agent)
            registry = all_servers()
            server_names = task.servers if task.servers is not None else tuple(registry)
            for name in server_names:
                if name not in registry:
                    raise KeyError(f"unknown server '{name}' for task '{task.id}'")
                await agent.connect_server(name, registry[name])
            agent.seed_classification_from_caps()

            answer = await agent.run(
                task.query,
                max_steps=task.max_steps,
                mode=task.mode,
                event_callback=events.append,
            )
            scored = await adapter.score(
                task, answer, CheckContext(workspace=workspace, agent=agent)
            )
        # A scorer may discover *during* the run that a prerequisite it must
        # provide isn't available (e.g. a toolchain it loads itself). It signals
        # that by returning a third element; absent it, the task simply ran.
        passed, detail = scored[0], scored[1]
        skipped = scored[2] if len(scored) > 2 else False
        tool_calls = sum(1 for e in events if e.get("type") == "tool_call")
        return RunResult(
            task.id, passed, detail, skipped=skipped,
            tool_calls=tool_calls, events=len(events),
        )
    except Exception:
        return RunResult(task.id, False, "run crashed", error=traceback.format_exc())
    finally:
        with contextlib.suppress(Exception):
            await agent.cleanup()
        os.chdir(orig_cwd)


async def run_benchmark(
    adapter: BenchmarkAdapter,
    *,
    model: str,
    backend: str = "vllm",
    report_path: str | None = None,
    limit: int | None = None,
    get_backend_override: Any | None = None,
    enforcement: str | None = None,
) -> dict:
    """Run *adapter*'s tasks through the agent; return a summary dict (optionally written).

    ``backend`` selects the live backend (``vllm``/``ollama``) via the ``LLM_BACKEND``
    env var the factory reads. ``get_backend_override`` injects a backend factory (e.g.
    a ``ScriptedBackend`` for CI), bypassing the live model. ``enforcement`` overrides
    the model-profile nudge level (``strict``/``light``/``off``) for every task, so a
    run is graded at one fixed level; ``None`` keeps each model's profile default.
    Pass-rate is computed over tasks that actually ran (skips excluded).
    """
    if get_backend_override is None:
        os.environ["LLM_BACKEND"] = backend
        from mimir.client.query_engine.backends.factory import clear_backend_cache
        clear_backend_cache()

    tasks = adapter.load_tasks(limit)
    started = time.time()
    results: list[RunResult] = []
    for task in tasks:
        results.append(
            await run_one(task, adapter, model, get_backend_override, enforcement)
        )

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
