"""Other agents run on the same tasks as MIMIR, at the same models, through the meter.

An :class:`ExternalAgent` says how to configure one agent (in a directory of the run's
own) and launch it on a task; :func:`run_external` runs it as ``run_one`` runs MIMIR —
toolchain check, workspace, timeout, the same adapter's grade — and reads its cost from
the meter's log by the key it was given (``Endpoint.run_tag``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimir.runner import (
    BatchAdapter, BenchmarkAdapter, BenchTask, CheckContext, RunResult, graded_result,
    missing_requirements,
)
from mimir.shared.secret_env import without_secrets


@dataclass
class Endpoint:
    """Where a run calls its model: the meter's base URL (``…/v1``), the served model's
    name, and the key that tags every call of this run in the meter's log."""

    base_url: str
    model: str
    run_tag: str


class ExternalAgent:
    """One agent compared with MIMIR. Subclasses name the executable and fill the hooks."""

    name = ""
    executable = ""

    def prepare(self, task: BenchTask, workspace: str, home: Path, endpoint: Endpoint) -> None:
        """Write what the agent reads (its config) under *home*, never in the workspace."""

    def command(self, task: BenchTask, workspace: str, home: Path,
                endpoint: Endpoint) -> list[str]:
        raise NotImplementedError

    def environment(self, home: Path, endpoint: Endpoint) -> dict[str, str]:
        """Variables added to the agent's environment (its home, its provider)."""
        return {}

    def answer(self, stdout: str, home: Path) -> str:
        """The agent's final answer: by default, what it printed."""
        return stdout.strip()


def meter_entries(log_path: str, run_tag: str) -> list[dict[str, Any]]:
    """The meter's records of the run tagged *run_tag*, in the order they were made."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    return [e for e in entries if e.get("run_tag") == run_tag]


def metered(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A run's cost as the meter saw it, in ``RunResult``'s fields."""
    def total(key: str) -> int | None:
        values = [e[key] for e in entries if isinstance(e.get(key), int)]
        return sum(values) if values else None

    curve = [e["prompt_tokens"] for e in entries if isinstance(e.get("prompt_tokens"), int)]
    return {"model_calls": len(entries), "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "cached_tokens": total("cached_tokens"), "prompt_curve": curve,
            "peak_prompt_tokens": max(curve, default=None)}


async def run_external(task: BenchTask, adapter: BenchmarkAdapter | BatchAdapter,
                       agent: ExternalAgent, endpoint: Endpoint, meter_log: str,
                       keep_workspace: bool = False) -> RunResult:
    """Run *agent* on *task* through the meter at *endpoint*, and grade it with *adapter*."""
    missing = missing_requirements(task)
    if agent.executable and shutil.which(agent.executable) is None:
        missing.append(agent.executable)
    if missing:
        return RunResult(task.id, False, f"skipped: missing {', '.join(missing)}", skipped=True,
                         variant=agent.name)
    workspace = tempfile.mkdtemp(prefix=f"bench_{task.id}_")
    home = Path(tempfile.mkdtemp(prefix=f"home_{agent.name}_"))
    started = time.monotonic()
    timed_out, stdout, error = False, "", ""
    try:
        if task.setup is not None:
            task.setup(workspace)
        agent.prepare(task, workspace, home, endpoint)
        # HOME stays (an agent installed for the user lives under it); what it reads and
        # writes of its own — config, data, state — moves to this run's home.
        env = {**without_secrets(os.environ),
               "XDG_CONFIG_HOME": str(home / ".config"),
               "XDG_DATA_HOME": str(home / ".local" / "share"),
               "XDG_STATE_HOME": str(home / ".local" / "state"),
               **agent.environment(home, endpoint)}
        try:
            proc = subprocess.run(agent.command(task, workspace, home, endpoint), cwd=workspace,
                                  env=env, capture_output=True, text=True,
                                  timeout=task.timeout_secs, stdin=subprocess.DEVNULL)
            stdout = proc.stdout
            if proc.returncode != 0:
                error = f"exit {proc.returncode}: {proc.stderr[-2000:]}"
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        answered = time.monotonic()
        answer = agent.answer(stdout, home)

        def measures() -> dict[str, Any]:
            return {**metered(meter_entries(meter_log, endpoint.run_tag)),
                    "elapsed_secs": round(time.monotonic() - started, 2),
                    "answer_secs": round(answered - started, 2), "variant": agent.name,
                    "run_tag": endpoint.run_tag}

        return await graded_result(adapter, task, answer,
                                   CheckContext(workspace=workspace, agent=None), measures,
                                   timed_out=timed_out, error=error)
    except Exception as exc:  # noqa: BLE001 — a failing run is recorded, not fatal
        return RunResult(task.id, False, "run crashed", error=f"{type(exc).__name__}: {exc}",
                         variant=agent.name, run_tag=endpoint.run_tag,
                         **metered(meter_entries(meter_log, endpoint.run_tag)))
    finally:
        shutil.rmtree(home, ignore_errors=True)
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def known_agents() -> dict[str, ExternalAgent]:
    """The agents MIMIR is compared with, by the name ``--agent`` takes."""
    from .agents import Copilot, MiniSweAgent, OpenCode

    return {a.name: a for a in (MiniSweAgent(), OpenCode(), Copilot())}
