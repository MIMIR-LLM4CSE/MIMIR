"""Tests for the headless run engine (``mimir.runner``).

Driven model-free with a ``ScriptedBackend`` (no live model, no MCP servers — the
test adapter uses ``servers=()``), mirroring the ScriptedBackend pattern in
``test_agent_loop.py``. Covers:

- report shape from ``run_benchmark`` (counts, pass_rate, per-task results);
- the unattended auto-approve hook (``_install_auto_approve``) approves any tool
  and disables the human continue prompt;
- per-task state isolation (each task gets a distinct workspace and a *fresh*
  agent, so nothing leaks between tasks);
- the skip path (a task with an unavailable ``requires`` is recorded skipped).
"""
from __future__ import annotations

import asyncio
import os
import unittest

from mimir.runner import BenchTask, CheckContext, run_benchmark
from mimir.runner.engine import _install_auto_approve
from mimir.tests._fake_backend import ScriptedBackend


def _scripted_override():
    backend = ScriptedBackend([])  # empty → every chat() ends the turn

    def _get_backend():
        return backend

    return _get_backend


class _RecordingAdapter:
    """Minimal adapter: two server-less tasks; records what each scorer saw."""

    name = "test_recording"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def load_tasks(self, limit=None):
        def _seed_a(ws: str) -> None:
            with open(os.path.join(ws, "from_task_a.txt"), "w") as fh:
                fh.write("a")

        tasks = [
            BenchTask(id="task_a", query="q", servers=(), max_steps=2, setup=_seed_a),
            BenchTask(id="task_b", query="q", servers=(), max_steps=2),
        ]
        return tasks[:limit] if limit is not None else tasks

    async def score(self, task, answer, ctx: CheckContext):
        self.seen.append({
            "task_id": task.id,
            "workspace": ctx.workspace,
            "agent_id": id(ctx.agent),
            # task_b must NOT see task_a's seeded file (workspace isolation):
            "leaked_a_file": os.path.isfile(os.path.join(ctx.workspace, "from_task_a.txt")),
        })
        return task.id == "task_a", f"scored {task.id}"


class RunBenchmarkReportTests(unittest.TestCase):
    def test_report_shape_and_per_task_results(self) -> None:
        adapter = _RecordingAdapter()
        summary = asyncio.run(run_benchmark(
            adapter, model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(summary["benchmark"], "test_recording")
        self.assertEqual(summary["backend"], "scripted")
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 1)        # only task_a passes
        self.assertEqual(summary["pass_rate"], 0.5)
        ids = [r["task_id"] for r in summary["results"]]
        self.assertEqual(ids, ["task_a", "task_b"])
        self.assertEqual([r["error"] for r in summary["results"]], ["", ""])

    def test_limit_caps_task_count(self) -> None:
        adapter = _RecordingAdapter()
        summary = asyncio.run(run_benchmark(
            adapter, model="dummy", limit=1, get_backend_override=_scripted_override(),
        ))
        self.assertEqual(summary["total"], 1)
        self.assertEqual([r["task_id"] for r in summary["results"]], ["task_a"])


class IsolationTests(unittest.TestCase):
    def test_fresh_agent_and_workspace_per_task(self) -> None:
        adapter = _RecordingAdapter()
        asyncio.run(run_benchmark(
            adapter, model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(len(adapter.seen), 2)
        a, b = adapter.seen
        # Distinct workspaces and distinct agent instances → no cross-task leakage.
        self.assertNotEqual(a["workspace"], b["workspace"])
        self.assertNotEqual(a["agent_id"], b["agent_id"])
        # task_b's workspace never received task_a's seed file.
        self.assertFalse(b["leaked_a_file"])

    def test_cwd_restored_after_run(self) -> None:
        before = os.getcwd()
        asyncio.run(run_benchmark(
            _RecordingAdapter(), model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(os.getcwd(), before)


class AutoApproveTests(unittest.TestCase):
    def test_install_auto_approve_approves_any_tool(self) -> None:
        class _Agent:
            allow_continue_prompt = True

            def _request_tool_approval(self, name, arguments, max_attempts=3):
                raise AssertionError("interactive approval must not be called")

        agent = _Agent()
        _install_auto_approve(agent)
        self.assertFalse(agent.allow_continue_prompt)
        # Any tool — including ones that would normally block (non-batch) — is approved.
        ok, reason = agent._request_tool_approval("bash_run", {"cmd": "ls"})
        self.assertTrue(ok)
        self.assertIn("auto-approve", reason)


class SkipTaskAdapter:
    """One task that requires a non-existent executable → must be skipped."""

    name = "test_skip"

    def load_tasks(self, limit=None):
        return [BenchTask(
            id="needs_bogus", query="q", servers=(), max_steps=2,
            requires=("definitely-not-a-real-binary-xyz",),
        )]

    async def score(self, task, answer, ctx):  # must never be called for a skipped task
        raise AssertionError("score() must not run for a skipped task")


class SkipSupportTests(unittest.TestCase):
    def test_missing_requirement_is_skipped_not_run(self) -> None:
        summary = asyncio.run(run_benchmark(
            SkipTaskAdapter(), model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["pass_rate"], 0.0)  # scored=0 → 0.0, no ZeroDivision
        self.assertTrue(summary["results"][0]["skipped"])


if __name__ == "__main__":
    unittest.main()
