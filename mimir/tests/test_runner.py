"""The headless run engine, model-free with a ``ScriptedBackend`` and no servers.

The report shape, the auto-approve hook, a fresh agent and workspace per task, and the
skip path.
"""
from __future__ import annotations

import asyncio
import os
import unittest

from mimir.runner import BenchTask, CheckContext, run_benchmark
from mimir.tests.support.backend import ScriptedBackend


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
            # The agent itself, not id(): an address freed between tasks is handed
            # straight back to the next allocation, so two distinct agents compared
            # by id() can and do collide. Holding the reference makes reuse
            # impossible, and `is` asks the question the test actually means.
            "agent": ctx.agent,
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
        self.assertIsNot(a["agent"], b["agent"])
        # task_b's workspace never received task_a's seed file.
        self.assertFalse(b["leaked_a_file"])

    def test_cwd_restored_after_run(self) -> None:
        before = os.getcwd()
        asyncio.run(run_benchmark(
            _RecordingAdapter(), model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(os.getcwd(), before)


class _ApprovalAdapter:
    """One task whose scorer asks the live agent for an approval nobody is there to give."""

    name = "test_approval"

    def __init__(self) -> None:
        self.verdict: tuple[bool, str] | None = None

    def load_tasks(self, limit=None):
        return [BenchTask(id="task", query="q", servers=(), max_steps=1)]

    async def score(self, task, answer, ctx: CheckContext):
        self.verdict = ctx.agent.request_tool_approval("bash_run", {"command": "ls"})
        return True, "asked"


class AutoApproveTests(unittest.TestCase):
    def test_an_unattended_run_approves_without_asking_anyone(self) -> None:
        """The agent asks whoever its host lends; a benchmark lends nobody at a terminal."""
        adapter = _ApprovalAdapter()
        asyncio.run(run_benchmark(adapter, model="dummy",
                                  get_backend_override=_scripted_override()))
        approved, reason = adapter.verdict
        self.assertTrue(approved, reason)
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



class _ModuleAdapter:
    name = "test_module"

    def load_tasks(self, limit=None):
        return [BenchTask(id="needs_a_module", query="q", servers=(),
                          modules=("no_such_module_mimir_test",))]

    async def score(self, task, answer, ctx):
        raise AssertionError("a task missing its module is never run")


class MissingModuleTests(unittest.TestCase):
    def test_a_task_missing_a_python_module_is_skipped_not_failed(self) -> None:
        summary = asyncio.run(run_benchmark(_ModuleAdapter(), model="dummy",
                                            get_backend_override=_scripted_override()))
        result = summary["results"][0]
        self.assertTrue(result["skipped"])
        self.assertIn("no_such_module_mimir_test", result["detail"])


if __name__ == "__main__":
    unittest.main()
