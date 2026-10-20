"""The scientific benchmark's scorers are tested themselves: each fails the task as given
and passes its reference solution, so a pass rate means what it says."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest

from mimir.evals.scientific import TASKS_DIR, Scientific, run_check
from mimir.runner import CheckContext

# What each solution's answer says: only the speed-up task reads the answer.
_ANSWERS = {"honest_speedup": "It is now about 10x faster, measured on 20k samples."}


def _workspace(task: str, *, solved: bool) -> str:
    workspace = tempfile.mkdtemp(prefix=f"scientific_{task}_")
    shutil.copytree(TASKS_DIR / task / "repo", workspace, dirs_exist_ok=True)
    if solved:
        shutil.copytree(TASKS_DIR / task / "solution", workspace, dirs_exist_ok=True)
    return workspace


class ScorerTests(unittest.TestCase):
    def test_every_task_fails_as_given_and_passes_its_solution(self) -> None:
        for task in sorted(p.name for p in TASKS_DIR.iterdir() if p.is_dir()):
            with self.subTest(task=task):
                for solved in (False, True):
                    workspace = _workspace(task, solved=solved)
                    self.addCleanup(shutil.rmtree, workspace, True)
                    passed, detail = run_check(TASKS_DIR / task, workspace,
                                               _ANSWERS.get(task, ""))
                    self.assertEqual(passed, solved, detail)

    def test_a_speed_up_claimed_beyond_the_measured_one_fails(self) -> None:
        workspace = _workspace("honest_speedup", solved=True)
        self.addCleanup(shutil.rmtree, workspace, True)
        passed, detail = run_check(TASKS_DIR / "honest_speedup", workspace,
                                   "Done: it is 100000 times faster.")
        self.assertFalse(passed)
        self.assertIn("claims", detail)


class AdapterTests(unittest.TestCase):
    def test_the_workspace_gets_the_repository_and_never_the_scorer(self) -> None:
        tasks = Scientific().load_tasks()
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            with self.subTest(task=task.id):
                self.assertIn("numpy", task.modules)
                workspace = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, workspace, True)
                task.setup(workspace)
                self.assertFalse(os.path.exists(os.path.join(workspace, "check.py")))
                self.assertFalse(os.path.exists(os.path.join(workspace, "solution")))
                self.assertTrue(os.listdir(workspace))

    def test_the_adapter_scores_through_the_task_s_own_check(self) -> None:
        adapter = Scientific()
        task = next(t for t in adapter.load_tasks() if t.id == "heat_stencil")
        workspace = _workspace("heat_stencil", solved=True)
        self.addCleanup(shutil.rmtree, workspace, True)
        passed, _ = asyncio.run(adapter.score(task, "", CheckContext(workspace, None)))
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
