"""The optimisation session on the code map: snapshots read back on either backend, and
each settled run turned into what it changed against the incumbent it was measured on."""

import json
import os
import tempfile
import unittest
from unittest import mock

from mimir.servers.proxy._lib import tree_snapshot
from mimir.servers.proxy._ops import code_map_steps


class _Workspace:
    """A tiny Python package, a shadow store beside it, and a way to snapshot it."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "ws")
        self.git = os.path.join(self.tmp.name, "store", "opt.git")
        os.makedirs(os.path.join(self.root, "pkg"))
        os.makedirs(os.path.dirname(self.git))
        for name, text in {"__init__.py": "", "a.py": "from . import b\n", "b.py": "", "c.py": ""}.items():
            self.write(name, text)

    def write(self, name: str, text: str) -> None:
        with open(os.path.join(self.root, "pkg", name), "w") as fh:
            fh.write(text)

    def snap(self, message: str) -> str:
        paths = [os.path.join(self.root, "pkg", n) for n in ("a.py", "b.py")]
        sid = tree_snapshot.snapshot(self.git, self.root, paths, message)
        assert sid
        return sid


class SnapshotReadTests(unittest.TestCase):
    def _check(self) -> None:
        ws = _Workspace()
        self.addCleanup(ws.tmp.cleanup)
        first = ws.snap("one")
        ws.write("a.py", "from . import c\n")
        second = ws.snap("two")
        self.assertEqual(tree_snapshot.listing(ws.git, ws.root, first), ["pkg/a.py", "pkg/b.py"])
        self.assertEqual(tree_snapshot.read(ws.git, ws.root, first, "pkg/a.py"), "from . import b\n")
        self.assertEqual(tree_snapshot.read(ws.git, ws.root, second, "pkg/a.py"), "from . import c\n")
        self.assertIsNone(tree_snapshot.read(ws.git, ws.root, first, "pkg/nope.py"))
        self.assertEqual(tree_snapshot.changed(ws.git, ws.root, first, second), ["pkg/a.py"])

    def test_a_git_snapshot_reads_back(self) -> None:
        if not tree_snapshot.git_available():
            self.skipTest("git is not installed")
        self._check()

    def test_a_copy_snapshot_reads_back(self) -> None:
        with mock.patch.object(tree_snapshot, "git_available", return_value=False):
            self._check()


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        ws = self.ws = _Workspace()
        self.addCleanup(ws.tmp.cleanup)
        base = ws.snap("baseline")
        ws.write("a.py", "from . import b\nfrom . import c\n")
        kept = ws.snap("r1")
        ws.write("a.py", "from . import c\n")
        ruled_out = ws.snap("r2")
        rows = [
            {"run_id": "r0", "launch_tree": base, "verdict": "accept", "primary_value": 10.0, "best_run_id": "r0"},
            {"run_id": "r1", "launch_tree": kept, "verdict": "accept", "primary_value": 8.0,
             "best_run_id": "r1", "axis": "import c"},
            {"run_id": "r2", "launch_tree": ruled_out, "verdict": "reject", "primary_value": 9.0,
             "best_run_id": "r1", "axis": "drop b"},
        ]
        ledger = os.path.join(ws.tmp.name, "ledger.jsonl")
        with open(ledger, "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        cfg = {"baseline_id": base, "baseline_run_id": "r0", "primary_metric": "time_s", "primary_goal": "min"}
        code_map_steps._CACHE.update({"key": None, "value": {}})
        patches = [
            mock.patch.object(code_map_steps, "resolve_proxy_name", return_value="solver"),
            mock.patch.object(code_map_steps, "opt_ledger_file", return_value=ledger),
            mock.patch.object(code_map_steps, "load_opt_config", return_value=cfg),
            mock.patch.object(code_map_steps, "read_ledger", return_value=rows),
            mock.patch.object(code_map_steps, "workspace_root", return_value=ws.root),
            mock.patch.object(code_map_steps, "opt_git_dir", return_value=ws.git),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.session = code_map_steps.session()

    def test_the_baseline_is_not_a_step(self) -> None:
        self.assertEqual(self.session["baseline"], {"run_id": "r0", "value": 10.0})
        self.assertEqual([s["run_id"] for s in self.session["steps"]], ["r1", "r2"])

    def test_a_kept_step_says_what_it_added_against_the_baseline(self) -> None:
        step = self.session["steps"][0]
        self.assertEqual((step["verdict"], step["axis"], step["changed"]), ("accept", "import c", ["pkg/a.py"]))
        self.assertAlmostEqual(step["delta"], -0.2)
        self.assertEqual([(e["from"], e["to"]) for e in step["added"]], [("pkg/a.py", "pkg/c.py")])
        self.assertEqual(step["removed"], [])

    def test_a_ruled_out_step_is_measured_against_the_kept_one(self) -> None:
        step = self.session["steps"][1]
        self.assertEqual(step["incumbent_run_id"], "r1")
        self.assertAlmostEqual(step["delta"], 0.125)
        self.assertEqual([(e["from"], e["to"]) for e in step["removed"]], [("pkg/a.py", "pkg/b.py")])
        self.assertEqual(step["added"], [])

    def test_no_session_is_no_steps(self) -> None:
        with mock.patch.object(code_map_steps, "resolve_proxy_name", return_value=""):
            self.assertEqual(code_map_steps.session(), {})


if __name__ == "__main__":
    unittest.main()
