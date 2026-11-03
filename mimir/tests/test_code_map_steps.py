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
            {"run_id": "r0", "launch_tree": base, "verdict": "accept", "primary_value": 10.0, "best_run_id": "r0",
             "peak_rss_mb": 600.0},
            {"run_id": "r1", "launch_tree": kept, "verdict": "accept", "primary_value": 8.0,
             "best_run_id": "r1", "axis": "import c", "peak_rss_mb": 900.0},
            {"run_id": "r2", "launch_tree": ruled_out, "verdict": "reject", "primary_value": 9.0,
             "best_run_id": "r1", "axis": "drop b"},
        ]
        ledger = os.path.join(ws.tmp.name, "ledger.jsonl")
        with open(ledger, "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        cfg = {"baseline_id": base, "baseline_run_id": "r0", "primary_metric": "time_s", "primary_goal": "min"}
        runs = os.path.join(ws.tmp.name, "runs")
        # The baseline and the kept run were profiled after their verdict; the ruled-out one was not.
        for run_id, files in {"r0": {"pkg/a.py": 0.5, "pkg/b.py": 0.25}, "r1": {"pkg/a.py": 0.75}}.items():
            os.makedirs(os.path.join(runs, run_id, "profile"))
            gpu = {"tool": "nsys", "busy": 0.5, "idle": 0.25, "memcpy": {"HtoD": 0.25},
                   "files": {"pkg/a.py": 0.5}, "kernels": {"k": {"share": 0.5, "calls": 4, "file": "pkg/a.py"}}}
            functions = {"pkg/a.py::solve": {"self": 0.5, "total": 0.75, "line": 3}}
            memory = {"tool": "massif", "peak_mb": 900.0, "files": {"pkg/a.py": 0.9}, "lines": {"pkg/a.py:3": 0.9}}
            with open(os.path.join(runs, run_id, "profile", "profile.json"), "w") as fh:
                json.dump({"tool": "py-spy", "samples": 400, "files": files, "outside": {"numpy": 0.25},
                           **({"gpu": gpu, "memory": memory, "functions": functions} if run_id == "r1" else {})}, fh)
        code_map_steps._CACHE.clear()
        patches = [
            mock.patch.object(code_map_steps, "resolve_proxy_name", return_value="solver"),
            mock.patch.object(code_map_steps, "opt_ledger_file", return_value=ledger),
            mock.patch.object(code_map_steps, "load_opt_config", return_value=cfg),
            mock.patch.object(code_map_steps, "read_ledger", return_value=rows),
            mock.patch.object(code_map_steps, "workspace_root", return_value=ws.root),
            mock.patch.object(code_map_steps, "opt_git_dir", return_value=ws.git),
            mock.patch.object(code_map_steps, "opt_session_runs_dir", return_value=runs),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.session = code_map_steps.session()

    def test_the_baseline_is_not_a_step(self) -> None:
        self.assertEqual({k: self.session["baseline"][k] for k in ("run_id", "value")},
                         {"run_id": "r0", "value": 10.0})
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

    def test_profiles_ride_in_fractions_of_the_baseline_s_time(self) -> None:
        """The kept run took 8 s against the baseline's 10: its shares count for 0.8 of the
        baseline's time, so the two runs' bars compare."""
        base = self.session["baseline"]["profile"]
        self.assertEqual((base["files"], base["outside"]), ({"pkg/a.py": 0.5, "pkg/b.py": 0.25}, 0.25))
        kept = self.session["steps"][0]["profile"]
        self.assertEqual((kept["files"], kept["outside"]), ({"pkg/a.py": 0.6}, 0.2))
        self.assertNotIn("gpu", base)
        self.assertNotIn("functions", base)
        self.assertEqual(kept["functions"], {"pkg/a.py::solve": {"self": 0.4, "total": 0.6, "line": 3}})
        self.assertEqual(kept["gpu"], {"tool": "nsys", "busy": 0.4, "idle": 0.2, "memcpy": {"HtoD": 0.2},
                                       "files": {"pkg/a.py": 0.4},
                                       "kernels": {"k": {"share": 0.4, "calls": 4, "file": "pkg/a.py"}}})
        self.assertEqual(kept["memory"], {"tool": "massif", "peak_mb": 900.0, "files": {"pkg/a.py": 0.9}})
        self.assertIsNone(self.session["steps"][1]["profile"])

    def test_a_step_carries_its_peak_memory_beside_its_incumbent_s(self) -> None:
        step = self.session["steps"][0]
        self.assertEqual((step["peak_rss_mb"], step["incumbent_peak_rss_mb"]), (900.0, 600.0))
        self.assertEqual(self.session["baseline"]["peak_rss_mb"], 600.0)
        self.assertIsNone(self.session["steps"][1]["peak_rss_mb"])

    def test_no_session_is_no_steps(self) -> None:
        with mock.patch.object(code_map_steps, "resolve_proxy_name", return_value=""):
            self.assertEqual(code_map_steps.session(), {})


class PickTests(unittest.TestCase):
    """The proxies the tab can switch between, and every proxy's steps together."""

    def setUp(self) -> None:
        def one(name: str, metric: str, goal: str, base: float, values: list[tuple[str, float, str]]) -> dict:
            steps, inc = [], base
            for i, (ts, v, verdict) in enumerate(values):
                steps.append({"n": i + 1, "proxy": name, "ts": ts, "run_id": f"{name}{i}", "axis": "", "verdict": verdict,
                              "value": v, "incumbent_value": inc, "changed": [], "added": [], "removed": []})
                inc = v if verdict == "accept" else inc
            return {"proxy": name, "metric": metric, "goal": goal, "baseline": {"run_id": "b", "value": base}, "steps": steps}

        self.sessions = {
            "solver": one("solver", "time_s", "min", 10.0, [("2026-01-01T00:00:00", 8.0, "accept"),
                                                            ("2026-01-03T00:00:00", 9.0, "reject")]),
            "mesh": one("mesh", "gflops", "max", 50.0, [("2026-01-02T00:00:00", 100.0, "accept")]),
        }
        for p in (mock.patch.object(code_map_steps, "proxies", return_value=["mesh", "solver"]),
                  mock.patch.object(code_map_steps, "resolve_proxy_name", side_effect=lambda n: n or "solver"),
                  mock.patch.object(code_map_steps, "session", side_effect=lambda n="": self.sessions[n or "solver"])):
            p.start()
            self.addCleanup(p.stop)

    def test_the_tab_is_told_the_proxies_and_the_active_one(self) -> None:
        out = code_map_steps.answer()
        self.assertEqual((out["proxy"], out["proxies"], out["active"], out["combined"]),
                         ("solver", ["mesh", "solver"], "solver", False))

    def test_another_proxy_s_steps_and_never_an_unknown_name(self) -> None:
        self.assertEqual(code_map_steps.answer("mesh")["proxy"], "mesh")
        self.assertEqual(code_map_steps.answer("../elsewhere")["proxy"], "solver")

    def test_every_proxy_in_the_order_they_settled_against_their_own_baseline(self) -> None:
        out = code_map_steps.answer(combined=True)
        self.assertTrue(out["combined"])
        self.assertEqual([(s["n"], s["proxy"]) for s in out["steps"]], [(1, "solver"), (2, "mesh"), (3, "solver")])
        # Lower is better for both: 8 s of 10 is 0.8, 100 GFLOP/s of 50 is 0.5.
        self.assertEqual([s["value"] for s in out["steps"]], [0.8, 0.5, 0.9])
        self.assertAlmostEqual(out["steps"][2]["delta"], 0.125)
        self.assertEqual(out["steps"][1]["metric"], "gflops")
        self.assertEqual(out["baseline"]["value"], 1.0)
        # The sessions themselves are left as they were cached.
        self.assertEqual(self.sessions["mesh"]["steps"][0]["n"], 1)

    def test_no_proxy_is_no_answer(self) -> None:
        with mock.patch.object(code_map_steps, "proxies", return_value=[]):
            self.assertEqual(code_map_steps.answer(combined=True), {})


if __name__ == "__main__":
    unittest.main()
