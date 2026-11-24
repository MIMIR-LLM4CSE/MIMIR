"""A benchmark of what distinguishes MIMIR from a generic coding agent: keeping a numerical
invariant, comparing to a tolerance, and not claiming what was not measured.

Each task is a directory under ``tasks/``: ``task.json`` (the query, its limits, the
modules it needs), ``repo/`` (what the workspace starts with), ``check.py`` (the scorer,
kept out of the workspace) and ``solution/`` (a reference fix, for the scorer's own tests).
Local, no cluster, numpy only.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mimir.runner import BenchTask, CheckContext

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
CHECK_TIMEOUT_SECS = 600


def run_check(task_dir: Path, workspace: str, answer: str) -> tuple[bool, str]:
    """Run a task's ``check.py`` on *workspace* and *answer*: ``(passed, detail)``."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(answer or "")
        answer_path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, str(task_dir / "check.py"), workspace, answer_path],
            capture_output=True, text=True, timeout=CHECK_TIMEOUT_SECS)
    except subprocess.TimeoutExpired:
        return False, f"the check did not finish within {CHECK_TIMEOUT_SECS}s"
    finally:
        os.unlink(answer_path)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        verdict = json.loads(lines[-1])
    except (IndexError, ValueError):
        return False, f"the check said nothing readable: {(proc.stderr or proc.stdout)[-400:]}"
    return bool(verdict.get("passed")), str(verdict.get("detail", ""))


class Scientific:
    """The scientific tasks, as a ``BenchmarkAdapter``."""

    name = "scientific"

    def load_tasks(self, limit: int | None = None) -> list[BenchTask]:
        tasks = []
        for task_dir in sorted(p for p in TASKS_DIR.iterdir() if (p / "task.json").is_file()):
            spec = json.loads((task_dir / "task.json").read_text())
            repo = task_dir / "repo"

            def _setup(workspace: str, repo: Path = repo) -> None:
                shutil.copytree(repo, workspace, dirs_exist_ok=True)

            tasks.append(BenchTask(
                id=task_dir.name, query=spec["query"], max_steps=spec.get("max_steps", 40),
                setup=_setup, requires=("python3",), modules=tuple(spec.get("modules", ())),
                meta={"dir": str(task_dir)},
            ))
        return tasks[:limit] if limit is not None else tasks

    async def score(self, task: BenchTask, answer: str, ctx: CheckContext) -> tuple[bool, str]:
        return run_check(Path(task.meta["dir"]), ctx.workspace, answer)
