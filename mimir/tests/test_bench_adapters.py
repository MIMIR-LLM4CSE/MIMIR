"""The adapters of published benchmarks, offline: SWE-bench's checkout, patch and grade
around a local repository, SciCode's solution file, expected values and step tests
around a small ``test_data.h5``."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimir.runner import CheckContext, RunResult
from mimir.evals.benchmarks.swebench import SWEBench

HAVE_H5 = all(importlib.util.find_spec(m) for m in ("h5py", "scipy", "numpy"))


def _git(cwd: str, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


class SWEBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        upstream = os.path.join(self.tmp, "upstream", "owner", "proj")
        os.makedirs(upstream)
        _git(upstream, "init", "--quiet")
        Path(upstream, "app.py").write_text("def add(a, b):\n    return a - b\n")
        _git(upstream, "add", "-A")
        _git(upstream, "commit", "--quiet", "-m", "base")
        self.base = _git(upstream, "rev-parse", "HEAD")
        Path(upstream, "app.py").write_text("def add(a, b):\n    return a + b\n")
        _git(upstream, "commit", "--quiet", "-am", "the fix")
        self.fix = _git(upstream, "rev-parse", "HEAD")
        self.adapter = SWEBench(work_dir=os.path.join(self.tmp, "work"),
                                repo_url=f"file://{self.tmp}/upstream/{{repo}}")
        self.row = {"instance_id": "owner__proj-1", "repo": "owner/proj",
                    "base_commit": self.base, "problem_statement": "add() subtracts.",
                    "hints_text": "look at app.py"}
        self.adapter.rows = lambda: [self.row, {**self.row, "instance_id": "owner__proj-2"}]

    def _workspace(self) -> str:
        task = self.adapter.load_tasks()[0]
        workspace = os.path.join(self.tmp, "ws")
        os.makedirs(workspace)
        task.setup(workspace)
        return workspace

    def test_the_workspace_holds_the_base_commit_and_nothing_later(self) -> None:
        workspace = self._workspace()
        self.assertIn("a - b", Path(workspace, "app.py").read_text())
        self.assertEqual(_git(workspace, "rev-parse", "HEAD"), self.base)
        self.assertEqual(_git(workspace, "rev-list", "--all", "--count"), "1")
        missing = subprocess.run(["git", "cat-file", "-e", self.fix], cwd=workspace,
                                 capture_output=True)
        self.assertNotEqual(missing.returncode, 0)

    def test_the_mirror_is_cloned_once_and_reused(self) -> None:
        self._workspace()
        mirror = self.adapter.mirror("owner/proj", self.base)
        self.assertTrue(Path(mirror, "HEAD").is_file())
        with patch("mimir.evals.benchmarks.swebench._git", side_effect=AssertionError("cloned again")):
            self.assertEqual(self.adapter.mirror("owner/proj", self.base), mirror)

    def test_the_patch_holds_every_change_new_files_included(self) -> None:
        workspace = self._workspace()
        Path(workspace, "app.py").write_text("def add(a, b):\n    return a + b\n")
        Path(workspace, "notes.py").write_text("X = 1\n")
        task = self.adapter.load_tasks()[0]
        artifact = asyncio.run(self.adapter.collect(task, "", CheckContext(workspace, None)))
        patch_text = artifact["model_patch"]
        self.assertIn("+    return a + b", patch_text)
        self.assertIn("+++ b/notes.py", patch_text)

    def test_instances_filter_and_an_unknown_one_is_refused(self) -> None:
        self.adapter.instances = ["owner__proj-2"]
        self.assertEqual([t.id for t in self.adapter.load_tasks()], ["owner__proj-2"])
        self.adapter.instances = ["nope"]
        with self.assertRaises(SystemExit):
            self.adapter.load_tasks()

    def test_hints_join_the_query_only_when_asked(self) -> None:
        self.assertNotIn("look at app.py", self.adapter.load_tasks()[0].query)
        self.adapter.hints = True
        self.assertIn("look at app.py", self.adapter.load_tasks()[0].query)

    def _runs(self) -> list[RunResult]:
        return [RunResult("owner__proj-1", False, "", pending=True,
                          artifact={"model_patch": "diff --git a/app.py b/app.py\n"}),
                RunResult("owner__proj-2", False, "", pending=True, artifact={"model_patch": ""})]

    def test_without_the_harness_the_runs_are_skipped_and_predictions_kept(self) -> None:
        with patch.object(SWEBench, "_unavailable", staticmethod(lambda: "no docker here")):
            verdicts = asyncio.run(self.adapter.grade(self._runs(), "lbl"))
        self.assertEqual({v[2] for v in verdicts.values()}, {True})
        self.assertIn("no docker here", verdicts["owner__proj-1"][1])
        lines = Path(self.tmp, "work", "runs", "lbl", "predictions.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(ln) for ln in lines][0],
                         {"instance_id": "owner__proj-1", "model_name_or_path": "mimir",
                          "model_patch": "diff --git a/app.py b/app.py\n"})

    def test_the_harness_report_gives_each_verdict(self) -> None:
        def harness(command, cwd, **kwargs):
            report = Path(cwd, "logs", "evaluation", "lbl", "results.json")
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"resolved_ids": ["owner__proj-1"],
                                          "empty_patch_ids": ["owner__proj-2"]}))
            self.command = command
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(SWEBench, "_unavailable", staticmethod(lambda: "")), \
                patch("mimir.evals.benchmarks.swebench.subprocess.run", side_effect=harness):
            verdicts = asyncio.run(self.adapter.grade(self._runs(), "lbl"))
        self.assertEqual(verdicts["owner__proj-1"], (True, "resolved"))
        self.assertFalse(verdicts["owner__proj-2"][0])
        self.assertIn("empty patch", verdicts["owner__proj-2"][1])
        self.assertIn("SWE-bench/SWE-bench_Lite", self.command)
        self.assertEqual(self.command[-2:], ["owner__proj-1", "owner__proj-2"])

    def test_a_harness_that_writes_no_report_fails_the_batch(self) -> None:
        with patch.object(SWEBench, "_unavailable", staticmethod(lambda: "")), \
                patch("mimir.evals.benchmarks.swebench.subprocess.run",
                      return_value=subprocess.CompletedProcess([], 1, "", "boom")):
            verdicts = asyncio.run(self.adapter.grade(self._runs(), "lbl"))
        self.assertIn("wrote no report (exit 1)", verdicts["owner__proj-1"][1])


@unittest.skipUnless(HAVE_H5, "needs h5py, scipy and numpy")
class SciCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        import h5py

        from mimir.evals.benchmarks.scicode import SciCode

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        with h5py.File(os.path.join(self.tmp, "test_data.h5"), "w") as fh:
            fh["99.1/test1/var1"] = [2.0, 4.0]
            fh["99.3/test1/var1"] = 3
        Path(self.tmp, "99.2.txt").write_text("def helper():\n    return 1\n")
        self.row = {
            "problem_id": "99", "problem_description_main": "Double things.",
            "problem_background_main": "", "problem_io": "x: array",
            "required_dependencies": "import numpy as np",
            "sub_steps": [
                {"step_number": "99.1", "step_description_prompt": "Double x.",
                 "step_background": "Doubling is x + x.", "ground_truth_code": "",
                 "function_header": "def double(x):\n    '''Twice x.'''",
                 "test_cases": ["assert np.allclose(double(np.array([1.0, 2.0])), target)"]},
                {"step_number": "99.2", "step_description_prompt": "A helper.",
                 "step_background": "", "ground_truth_code": "",
                 "function_header": "def helper():\n    '''One.'''", "test_cases": []},
                {"step_number": "99.3", "step_description_prompt": "Hold v.",
                 "step_background": "", "ground_truth_code": "",
                 "function_header": "class Box:\n    def __init__(self, v):\n        '''Hold v.'''",
                 "test_cases": ["assert Box(3).v == target + helper() - 1"]},
            ]}
        self.adapter = SciCode(data_dir=self.tmp)
        self.adapter.rows = lambda: [self.row]

    def _workspace(self):
        task = self.adapter.load_tasks()[0]
        workspace = os.path.join(self.tmp, "ws")
        os.makedirs(workspace)
        task.setup(workspace)
        return task, workspace

    def test_the_solution_file_declares_every_step_and_gives_the_provided_one(self) -> None:
        task, workspace = self._workspace()
        source = Path(workspace, "solution.py").read_text()
        compile(source, "solution.py", "exec")
        self.assertIn("        raise NotImplementedError", source)   # the method's own indent
        self.assertIn("# Step 99.2 (provided)\ndef helper():\n    return 1", source)
        self.assertEqual([s["step"] for s in task.meta["steps"]], ["99.1", "99.3"])
        self.assertIn("### Step 99.2 (provided)", task.query)
        self.assertNotIn("Doubling is", task.query)
        self.adapter.background = True
        self.assertIn("Doubling is", self.adapter.load_tasks()[0].query)

    def test_the_stubs_fail_and_a_solution_passes_every_step(self) -> None:
        task, workspace = self._workspace()
        ctx = CheckContext(workspace, None)
        passed, detail = asyncio.run(self.adapter.score(task, "", ctx))
        self.assertFalse(passed)
        self.assertIn("sub-steps 0/2 passed", detail)
        self.assertIn("NotImplementedError", detail)
        source = Path(workspace, "solution.py").read_text()
        source = source.replace("'''Twice x.'''\n    raise NotImplementedError",
                                "'''Twice x.'''\n    return 2 * x")
        source = source.replace("'''Hold v.'''\n        raise NotImplementedError",
                                "'''Hold v.'''\n        self.v = v")
        Path(workspace, "solution.py").write_text(source)
        self.assertEqual(asyncio.run(self.adapter.score(task, "", ctx)),
                         (True, "sub-steps 2/2 passed"))

    def test_without_the_expected_values_nothing_runs(self) -> None:
        os.unlink(os.path.join(self.tmp, "test_data.h5"))
        with self.assertRaises(SystemExit):
            self.adapter.load_tasks()

    def test_a_step_without_tests_or_its_shipped_code_is_refused(self) -> None:
        os.unlink(os.path.join(self.tmp, "99.2.txt"))
        with self.assertRaises(SystemExit) as raised:
            self.adapter.load_tasks()
        self.assertIn("99.2.txt", str(raised.exception))

    def test_the_expected_values_keep_their_shapes(self) -> None:
        import h5py
        import numpy as np

        from mimir.evals.benchmarks.scicode.targets import targets

        path = os.path.join(self.tmp, "shapes.h5")
        with h5py.File(path, "w") as fh:
            fh["1.1/test1/var1"] = 1.5
            fh["1.1/test1/var2"] = np.bytes_(b"text")
            fh["1.1/test2/var1/list/0"] = 7
            fh["1.1/test2/var1/list/1"] = 8
            fh["1.1/test3/var1/0.5"] = 2
            fh["1.1/test3/var1/name"] = np.bytes_(b"x")
            sparse = fh.create_group("1.1/test4/var1/sparse_matrix")
            sparse["data"], sparse["shape"] = [1.0], [2, 2]
            sparse["row"], sparse["col"] = [0], [1]
        values = targets(path, "1.1", 4)
        self.assertEqual(values[0], (1.5, "text"))
        self.assertEqual(values[1], [7, 8])
        self.assertEqual(values[2], {0.5: 2, "name": "x"})
        self.assertEqual(values[3].toarray().tolist(), [[0.0, 1.0], [0.0, 0.0]])
        with self.assertRaises(KeyError):
            targets(path, "1.1", 5)


if __name__ == "__main__":
    unittest.main()
