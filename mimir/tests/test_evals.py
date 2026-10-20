"""The evaluation loop's plumbing, model-free: runs interleaved under variants, a report
written as it grows, and a comparison that says when a gap is not measurable."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import unittest

from mimir.evals.__main__ import _run
from mimir.evals.report import compare, load_runs
from mimir.runner import BenchTask
from mimir.tests.support.backend import ScriptedBackend


class TwoTasks:
    """Two server-less tasks; the first passes, the second does not."""

    name = "two_tasks"

    def load_tasks(self, limit=None):
        tasks = [BenchTask(id="a", query="q", servers=(), max_steps=1),
                 BenchTask(id="b", query="q", servers=(), max_steps=1)]
        return tasks[:limit] if limit is not None else tasks

    async def score(self, task, answer, ctx):
        return task.id == "a", f"scored {task.id}"


def _runs(variant: str, passed: int, n: int) -> list[dict]:
    return [{"task_id": "t", "variant": variant, "passed": i < passed, "skipped": False,
             "tool_calls": 3, "model_calls": 4, "prompt_tokens": None, "nudges": {},
             "incomplete": False} for i in range(n)]


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.variant = os.path.join(self.dir, "terse.json")
        with open(self.variant, "w") as fh:
            json.dump({"sections": {"validation": "## Validation\nCheck."}}, fh)

    def test_every_task_runs_under_every_variant_and_the_report_holds_them(self) -> None:
        out = os.path.join(self.dir, "report.json")
        args = argparse.Namespace(adapter="mimir.tests.test_evals:TwoTasks", variant=[self.variant],
                                  repeat=2, limit=None, model="dummy", backend="vllm",
                                  enforcement=None, out=out)
        asyncio.run(_run(args, get_backend_override=lambda: ScriptedBackend([])))
        runs = load_runs([out])
        self.assertEqual(len(runs), 8)
        # Interleaved: each task under both variants before the next task.
        self.assertEqual([(r["task_id"], r["variant"]) for r in runs[:4]],
                         [("a", "code"), ("a", "terse"), ("b", "code"), ("b", "terse")])
        self.assertEqual({r["repetition"] for r in runs}, {0, 1})
        table = compare(runs)
        self.assertIn("| code | 50 % (2/4)", table)
        self.assertIn("**terse** against **code**: +0 points", table)
        self.assertIn("no measurable difference", table)


class CompareTests(unittest.TestCase):
    def test_a_gap_whose_interval_holds_zero_is_not_called_a_difference(self) -> None:
        table = compare(_runs("code", 6, 10) + _runs("terse", 7, 10))
        self.assertIn("no measurable difference", table)

    def test_a_clear_gap_is_called_in_its_direction(self) -> None:
        self.assertIn("measurably worse", compare(_runs("code", 18, 20) + _runs("terse", 6, 20)))
        self.assertIn("measurably better", compare(_runs("code", 6, 20) + _runs("terse", 18, 20)))

    def test_skipped_runs_are_not_counted(self) -> None:
        runs = _runs("code", 1, 2) + [{**_runs("code", 0, 1)[0], "skipped": True}]
        self.assertIn("50 % (1/2)", compare(runs))


if __name__ == "__main__":
    unittest.main()
