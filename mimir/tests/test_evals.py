"""The evaluation loop's plumbing, model-free: runs interleaved under variants, a report
written as it grows, and a comparison that says when a gap is not measurable."""
from __future__ import annotations

import argparse
import asyncio
import importlib
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


class Configured(TwoTasks):
    """TwoTasks taking keyword arguments, as ``--adapter-arg`` passes them."""

    made_with: dict = {}

    def __init__(self, **kwargs) -> None:
        Configured.made_with = kwargs


class Batched:
    """Two tasks graded together; every batch it was given is recorded on the class."""

    name = "batched"
    batches: list[tuple[str, list[str]]] = []

    def load_tasks(self, limit=None):
        return [BenchTask(id="a", query="q", servers=(), max_steps=1),
                BenchTask(id="b", query="q", servers=(), max_steps=1)]

    async def collect(self, task, answer, ctx):
        return {"patch": f"diff of {task.id}"}

    async def grade(self, runs, label):
        type(self).batches.append((label, [r.artifact["patch"] for r in runs]))
        return {r.task_id: (r.task_id == "a", "graded") for r in runs}


def _as_loaded(name: str):
    """The class as ``--adapter`` loads it: pytest may import this file under another name."""
    return getattr(importlib.import_module("mimir.tests.test_evals"), name)


def _args(out: str, **kw) -> argparse.Namespace:
    return argparse.Namespace(**{"adapter": "mimir.tests.test_evals:TwoTasks", "adapter_arg": [],
                                 "variant": [], "repeat": 1, "limit": None, "model": "dummy",
                                 "backend": "vllm", "enforcement": None, "out": out,
                                 "resume": False, "keep_workspaces": False, "agent": [],
                                 "meter": None, "meter_log": None, **kw})


def _scripted():
    return ScriptedBackend([])


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
        asyncio.run(_run(_args(out, variant=[self.variant], repeat=2), get_backend_override=_scripted))
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


    def test_adapter_args_reach_the_adapter_and_the_report(self) -> None:
        out = os.path.join(self.dir, "report.json")
        asyncio.run(_run(_args(out, adapter="mimir.tests.test_evals:Configured",
                               adapter_arg=["dataset=verified", "instances=a,b"]),
                         get_backend_override=_scripted))
        made_with = _as_loaded("Configured").made_with
        self.assertEqual(made_with, {"dataset": "verified", "instances": "a,b"})
        with open(out) as fh:
            self.assertEqual(json.load(fh)["adapter_args"], made_with)

    def test_an_adapter_arg_without_a_value_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            asyncio.run(_run(_args(os.path.join(self.dir, "r.json"), adapter_arg=["dataset"]),
                             get_backend_override=_scripted))

    def test_a_batch_adapter_is_graded_once_per_repetition_and_variant(self) -> None:
        batched = _as_loaded("Batched")
        batched.batches = []
        out = os.path.join(self.dir, "report.json")
        asyncio.run(_run(_args(out, adapter="mimir.tests.test_evals:Batched",
                               variant=[self.variant], repeat=2),
                         get_backend_override=_scripted))
        labels = [label for label, _ in batched.batches]
        self.assertEqual(len(labels), 4)
        self.assertEqual([lb.split("-")[1:3] for lb in labels],
                         [["code", "r0"], ["terse", "r0"], ["code", "r1"], ["terse", "r1"]])
        self.assertEqual(batched.batches[0][1], ["diff of a", "diff of b"])
        runs = load_runs([out])
        self.assertFalse(any(r["pending"] for r in runs))
        self.assertEqual(sum(r["passed"] for r in runs), 4)
        self.assertEqual(runs[0]["artifact"], {"patch": "diff of a"})

    def test_a_resumed_report_keeps_its_runs_and_grades_what_was_pending(self) -> None:
        out = os.path.join(self.dir, "report.json")
        asyncio.run(_run(_args(out, adapter="mimir.tests.test_evals:Batched"),
                         get_backend_override=_scripted))
        with open(out) as fh:
            data = json.load(fh)
        # Interrupted after the first run, before the batch was graded.
        first = {**data["runs"][0], "pending": True, "passed": False}
        data["runs"] = [first]
        with open(out, "w") as fh:
            json.dump(data, fh)
        batched = _as_loaded("Batched")
        batched.batches = []
        asyncio.run(_run(_args(out, adapter="mimir.tests.test_evals:Batched", resume=True),
                         get_backend_override=_scripted))
        self.assertEqual(batched.batches[0][1], ["diff of a", "diff of b"])
        runs = load_runs([out])
        self.assertEqual([(r["task_id"], r["passed"]) for r in runs], [("a", True), ("b", False)])

    def test_a_report_made_with_other_options_is_not_resumed(self) -> None:
        out = os.path.join(self.dir, "report.json")
        asyncio.run(_run(_args(out), get_backend_override=_scripted))
        with self.assertRaises(SystemExit):
            asyncio.run(_run(_args(out, model="other", resume=True), get_backend_override=_scripted))


class CompareTests(unittest.TestCase):
    def test_a_gap_whose_interval_holds_zero_is_not_called_a_difference(self) -> None:
        table = compare(_runs("code", 6, 10) + _runs("terse", 7, 10))
        self.assertIn("no measurable difference", table)

    def test_a_clear_gap_is_called_in_its_direction(self) -> None:
        self.assertIn("measurably worse", compare(_runs("code", 18, 20) + _runs("terse", 6, 20)))
        self.assertIn("measurably better", compare(_runs("code", 6, 20) + _runs("terse", 18, 20)))

    def test_several_variants_at_once_widen_every_interval(self) -> None:
        """A gap measurable against one wording alone is not, among five tried together."""
        alone = compare(_runs("code", 10, 20) + _runs("v1", 17, 20))
        self.assertIn("**v1** against **code**: +35 points", alone)
        self.assertIn("measurably better", alone)
        together = compare(_runs("code", 10, 20) + _runs("v1", 17, 20)
                           + sum((_runs(f"v{i}", 10, 20) for i in range(2, 6)), []))
        self.assertIn("5 variants are compared with the reference at once", together)
        self.assertIn("joint 95 % interval", together)
        self.assertNotIn("measurably better", together)

    def test_runs_of_the_same_tasks_are_also_compared_pairwise(self) -> None:
        reference = [{**_runs("code", 0, 1)[0], "task_id": f"t{i}", "repetition": 0}
                     for i in range(6)]
        better = [{**r, "variant": "terse", "passed": True} for r in reference]
        table = compare(reference + better)
        self.assertIn("Paired over 6 runs of the same tasks: 6 passed only under **terse**", table)
        self.assertIn("exact McNemar p = 0.031, measurable", table)

    def test_what_each_variant_cost_is_tabled(self) -> None:
        runs = [{**_runs("code", 1, 1)[0], "prompt_tokens": 1000, "cached_tokens": 750,
                 "completion_tokens": 50, "peak_prompt_tokens": 600, "answer_secs": 12.0,
                 "side_calls": {"compaction": 1}, "context_events": {"compact": 1},
                 "refused": [{"tool": "bash_run"}]}]
        table = compare(runs)
        self.assertIn("## Cost per run", table)
        self.assertIn("| code | 4.0 | 1.0 | 1000.0 | 50.0 | 75 % | 600.0 | 12.0 | 1.0 | 1.0 |",
                      table)

    def test_skipped_runs_are_not_counted(self) -> None:
        runs = _runs("code", 1, 2) + [{**_runs("code", 0, 1)[0], "skipped": True}]
        self.assertIn("50 % (1/2)", compare(runs))


if __name__ == "__main__":
    unittest.main()
