"""Merging proxies: each one's kept patch, the conflicts between them, the gain they
promise together, and the merge measured all at once, then one at a time when the whole
falls short."""

import json
import os
from unittest import mock

from mimir.servers.proxy._lib import layout
from mimir.servers.proxy._ops import code_map_steps, merge
from mimir.shared.atomic import write_json
from mimir.tests.support.proxy import (
    _TmpStorageTest,
    _eval,
    eval_session,
    procs,
    server_proxy,
    tree_snapshot,
)

ORIGINAL = {"a.py": "A = 1\n\n\n\n\nX = 0\n", "b.py": "B = 1\n", "c.py": "C = 1\n"}


class _MergeFixture(_TmpStorageTest):
    """Three proxies optimised one after the other in one tree, each from the original:
    pa edits a.py's first line (0.8 alone), pb edits b.py (0.9), pc edits a.py's first
    line otherwise (0.85), so pa and pc conflict."""

    def setUp(self) -> None:
        super().setUp()
        if not tree_snapshot.git_available():
            self.skipTest("git is not installed")
        self.harness = os.path.join(self.root, "harness.py")
        with open(self.harness, "w") as fh:
            fh.write("import a\n")
        self._write(ORIGINAL)
        self.n = 0
        for name, edit, value in (("pa", {"a.py": "A = 2\n\n\n\n\nX = 0\n"}, 8.0),
                                  ("pb", {"b.py": "B = 2\n"}, 9.0),
                                  ("pc", {"a.py": "A = 3\n\n\n\n\nX = 0\n"}, 8.5)):
            self._source(name, edit, value)
            self._write(ORIGINAL)

    def _write(self, files: dict) -> None:
        for rel, text in files.items():
            with open(os.path.join(self.root, rel), "w") as fh:
                fh.write(text)

    def _read(self, rel: str) -> str:
        with open(os.path.join(self.root, rel)) as fh:
            return fh.read()

    def _source(self, name: str, edit: dict, value: float) -> None:
        exe = os.path.join(self.root, "tiny.py")
        with open(exe, "w") as fh:
            fh.write("print('PROXY_METRICS_BEGIN')\nprint('time_s=0.1')\nprint('PROXY_METRICS_END')\n")
        server_proxy.proxy_manage(op="register", name=name, executable_path=exe,
                                  run_cmd_template="python3 {executable}", confirm=True)
        server_proxy.proxy_manage(op="suite_define", name="bench",
                                  cases=[{"case_id": "a", "proxy_name": "pa"}], confirm=True)
        res = _eval(op="init", proxy_name=name, benchmark_name="bench",
                    requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
                    proxy_source_path=self.harness,
                    optimize_paths=[os.path.join(self.root, f) for f in ORIGINAL], confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        self._settle(name, 10.0)
        self._write(edit)
        self.assertEqual(self._settle(name, value)["verdict"], "accept")

    def _complete(self, name: str, value: float, passed: bool = True) -> None:
        """A finished run of *name* on the tree as it stands, as the runner leaves it."""
        self.n += 1
        run_dir = os.path.join(layout.opt_session_runs_dir(name), f"20240101T{self.n:06d}Z")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump({"all_passed": passed, "best_time_s": value if passed else None,
                       "results": [{"metrics": {"time_s": value}}]}, fh)
        with open(procs.run_log_path(run_dir), "w") as fh:
            fh.write(f"[proxy_runner] summary cases_passed=1 cases_total=1 all_passed={passed} "
                     f"best_case=a best_time_s={value}\n")
        cfg = eval_session.load_opt_config(name)
        paths = cfg["optimize_paths"]
        sid = tree_snapshot.snapshot(eval_session.opt_git_dir(), eval_session.workspace_root(), paths, "launch")
        write_json(os.path.join(run_dir, "tree_at_launch.json"),
                   {"snapshot_id": sid, "paths": paths, "is_baseline": sid == cfg.get("baseline_id")})
        procs.update_opt_active_link(name, run_dir)

    def _settle(self, name: str, value: float) -> dict:
        self._complete(name, value)
        return server_proxy.proxy_eval_status(op="results", proxy_name=name)

    def _merge(self) -> dict:
        with mock.patch.object(merge, "_in_copy", return_value=True):
            res = server_proxy.proxy_merge(op="start", proxies=["*"], proxy_name="merge-t", confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        nxt = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertIn("merge: baseline", nxt["next_step"])
        self._complete("merge-t", 10.0)
        return server_proxy.proxy_merge(op="next", confirm=True)


class PlanTests(_MergeFixture):
    def test_plan_orders_the_sources_and_names_their_conflict(self) -> None:
        res = server_proxy.proxy_merge(op="plan", proxies=["*"])
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["order"], ["pa", "pc", "pb"])
        self.assertAlmostEqual(res["expected"]["relative"], 0.8 * 0.85 * 0.9, places=5)
        pair = next(p for p in res["pairs"] if {p["a"], p["b"]} == {"pa", "pc"})
        self.assertEqual(pair["shared"], ["a.py"])
        self.assertEqual(pair["conflicts"], ["a.py"])
        apart = next(p for p in res["pairs"] if {p["a"], p["b"]} == {"pa", "pb"})
        self.assertEqual((apart["shared"], apart["conflicts"]), ([], []))
        self.assertIn("spawn_agent", res["next_step"])

    def test_a_proxy_with_nothing_kept_is_set_aside(self) -> None:
        server_proxy.proxy_manage(op="register", name="pd", executable_path=self.harness,
                                  run_cmd_template="python3 {executable}", confirm=True)
        _eval(op="init", proxy_name="pd", benchmark_name="bench",
              requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
              proxy_source_path=self.harness, optimize_paths=[os.path.join(self.root, "c.py")], confirm=True)
        res = server_proxy.proxy_merge(op="plan", proxies=["pa", "pd"])
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["refused"][0]["proxy"], "pd")

    def test_another_metric_is_refused(self) -> None:
        cfg = eval_session.load_opt_config("pb")
        cfg["primary_metric"] = "gflops"
        eval_session.save_opt_config(cfg)
        res = server_proxy.proxy_merge(op="plan", proxies=["pa", "pb"])
        self.assertEqual(res["status"], "error")
        self.assertIn("metric", res["error"])

    def test_start_is_refused_in_the_main_workspace(self) -> None:
        res = server_proxy.proxy_merge(op="start", proxies=["*"], confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("plan", res["hint"])


class MergeRunTests(_MergeFixture):
    def test_the_whole_holds_its_promise(self) -> None:
        res = self._merge()
        self.assertIn("merge: all (pa + pb)", res["next_step"])
        self.assertEqual((self._read("a.py"), self._read("b.py")), ("A = 2\n\n\n\n\nX = 0\n", "B = 2\n"))
        self._complete("merge-t", 7.2)
        rep = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(rep["phase"], "done")
        self.assertEqual(rep["final"]["from"], "all")
        self.assertEqual(rep["kept"], ["pa", "pb"])
        self.assertEqual(rep["set_aside"], [{"proxy": "pc", "why": "conflict"}])
        self.assertAlmostEqual(rep["final"]["gain"], 0.28, places=5)
        self.assertTrue(rep["all"]["holds"])

    def test_a_short_whole_is_measured_one_at_a_time(self) -> None:
        self._merge()
        self._complete("merge-t", 9.5)
        res = server_proxy.proxy_merge(op="next", confirm=True)
        # Back to the original, the best alone first.
        self.assertIn("merge: + pa", res["next_step"])
        self.assertEqual((self._read("a.py"), self._read("b.py")), ("A = 2\n\n\n\n\nX = 0\n", "B = 1\n"))
        self._complete("merge-t", 8.0)
        res = server_proxy.proxy_merge(op="next", confirm=True)
        # pc conflicts with what was kept; pb is next.
        self.assertIn("merge: + pb", res["next_step"])
        self._complete("merge-t", 8.5)
        rep = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(rep["final"]["from"], "chain")
        self.assertEqual(rep["kept"], ["pa"])
        why = {s["proxy"]: s["why"] for s in rep["set_aside"]}
        self.assertEqual(why, {"pc": "conflict", "pb": "interference"})
        self.assertEqual(self._read("b.py"), "B = 1\n")
        pb = next(s for s in rep["steps"] if s["proxy"] == "pb")
        self.assertGreater(pb["interaction"], 1)

    def test_next_waits_for_the_run_it_named(self) -> None:
        res = self._merge()
        again = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(again["next_step"], res["next_step"])

    def test_the_map_draws_the_merge_and_keeps_it_frozen(self) -> None:
        self._merge()
        self._complete("merge-t", 7.2)
        server_proxy.proxy_merge(op="next", confirm=True)
        drawn = code_map_steps.answer("merge-t")
        self.assertEqual(drawn["merge"]["phase"], "done")
        meta = {m["name"]: m for m in drawn["proxies_meta"]}
        self.assertTrue(meta["merge-t"]["merge"])
        self.assertEqual(meta["merge-t"]["sources"], ["pa", "pc", "pb"])
        self.assertAlmostEqual(meta["pa"]["gain"], 0.2, places=5)
        key = code_map_steps.freeze("merge-t", "cleaned")
        self.assertEqual(code_map_steps.session(key)["merge"]["phase"], "done")

    def test_ticked_proxies_bring_their_forecast(self) -> None:
        drawn = code_map_steps.answer(selected=["pa", "pb"])
        self.assertTrue(drawn["combined"])
        self.assertEqual(drawn["selected"], ["pa", "pb"])
        self.assertAlmostEqual(drawn["forecast"]["expected"], 0.72, places=5)
