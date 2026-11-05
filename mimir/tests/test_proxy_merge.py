"""Merging proxies: each one's kept patch, the conflicts between them, the gain they
promise together, and the merge measured all at once, then one at a time when the whole
falls short."""

import json
import os
import subprocess
import unittest
from unittest import mock

from mimir.servers.proxy._lib import layout, store
from mimir.servers.proxy._ops import merge, merge_sources
from mimir.servers.proxy._ops.code_map import session, tab
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
    """Three proxies optimised one after the other in one git tree, each from the original
    and each on a suite of its own: pa edits a.py's first line (0.8 alone), pb edits b.py
    (0.9), pc edits a.py's first line otherwise (0.85), so pa and pc conflict."""

    def setUp(self) -> None:
        super().setUp()
        if not tree_snapshot.git_available():
            self.skipTest("git is not installed")
        self.harness = os.path.join(self.root, "harness.py")
        with open(self.harness, "w") as fh:
            fh.write("import a\n")
        with open(os.path.join(self.root, "tiny.py"), "w") as fh:
            fh.write("print('PROXY_METRICS_BEGIN')\nprint('time_s=0.1')\nprint('PROXY_METRICS_END')\n")
        self._write(ORIGINAL)
        for argv in (["init", "-q"], ["add", "a.py", "b.py", "c.py", "harness.py", "tiny.py"],
                     ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "original"]):
            subprocess.run(["git", *argv], cwd=self.root, check=True, capture_output=True)
        self.n = 0
        for name, edit, value in (("pa", {"a.py": "A = 2\n\n\n\n\nX = 0\n"}, 8.0),
                                  ("pb", {"b.py": "B = 2\n"}, 9.0),
                                  ("pc", {"a.py": "A = 3\n\n\n\n\nX = 0\n"}, 8.5)):
            self._source(name, edit, value)
            self._write(ORIGINAL)

    def _git(self, *argv: str) -> str:
        return subprocess.run(["git", *argv], cwd=self.root, capture_output=True, text=True).stdout

    def _write(self, files: dict) -> None:
        for rel, text in files.items():
            with open(os.path.join(self.root, rel), "w") as fh:
                fh.write(text)

    def _read(self, rel: str) -> str:
        with open(os.path.join(self.root, rel)) as fh:
            return fh.read()

    def _source(self, name: str, edit: dict, value: float) -> None:
        exe = os.path.join(self.root, "tiny.py")
        server_proxy.proxy_manage(op="register", name=name, executable_path=exe,
                                  run_cmd_template="python3 {executable}", confirm=True)
        server_proxy.proxy_manage(op="suite_define", name=f"bench_{name}",
                                  cases=[{"case_id": "a", "proxy_name": name}], confirm=True)
        res = _eval(op="init", proxy_name=name, benchmark_name=f"bench_{name}",
                    requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
                    proxy_source_path=self.harness,
                    optimize_paths=[os.path.join(self.root, f) for f in ORIGINAL], confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        self._settle(name, 10.0)
        self._write(edit)
        self.assertEqual(self._settle(name, value)["verdict"], "accept")

    def _complete(self, name: str, value, passed: bool = True) -> None:
        """A finished run of *name* on the tree as it stands, as the runner leaves it:
        *value* is one case's time, or {case: time}."""
        self.n += 1
        run_dir = os.path.join(layout.opt_session_runs_dir(name), f"20240101T{self.n:06d}Z")
        os.makedirs(run_dir, exist_ok=True)
        times = value if isinstance(value, dict) else {"a": value}
        total = sum(times.values())
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump({"all_passed": passed, "best_time_s": min(times.values()) if passed else None,
                       "total_time_s": total if passed else None,
                       "results": [{"case_id": c, "passed": passed, "metrics": {"time_s": t}}
                                   for c, t in times.items()]}, fh)
        with open(procs.run_log_path(run_dir), "w") as fh:
            fh.write(f"[proxy_runner] summary cases_passed={len(times)} cases_total={len(times)} "
                     f"all_passed={passed} best_case=a best_time_s={min(times.values())}\n")
        cfg = eval_session.load_opt_config(name)
        paths = cfg["optimize_paths"]
        sid = tree_snapshot.snapshot(eval_session.opt_git_dir(), eval_session.workspace_root(), paths, "launch")
        write_json(os.path.join(run_dir, "tree_at_launch.json"),
                   {"snapshot_id": sid, "paths": paths, "is_baseline": sid == cfg.get("baseline_id")})
        procs.update_opt_active_link(name, run_dir)

    def _settle(self, name: str, value) -> dict:
        self._complete(name, value)
        return server_proxy.proxy_eval_status(op="results", proxy_name=name)

    def _merge(self) -> dict:
        """Started and its baseline measured: answers what next says then."""
        with mock.patch.object(merge, "_in_copy", return_value=True):
            res = server_proxy.proxy_merge(op="start", proxies=["*"], proxy_name="merge-t", confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        nxt = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertIn("merge: baseline", nxt["next_step"])
        self._complete("merge-t", {"pa/a": 10.0, "pc/a": 10.0, "pb/a": 10.0})
        return server_proxy.proxy_merge(op="next", confirm=True)

    def _resolve_keeping_pa(self) -> dict:
        self._write({"a.py": "A = 2\n\n\n\n\nX = 0\n"})
        return server_proxy.proxy_merge(op="resolved", resolution="pa's rewrite kept: method choice",
                                        confirm=True)


class PlanTests(_MergeFixture):
    def test_plan_takes_proxies_of_other_suites_and_names_their_conflict(self) -> None:
        res = server_proxy.proxy_merge(op="plan", proxies=["*"])
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["order"], ["pa", "pc", "pb"])
        self.assertEqual({s["suite"] for s in res["sources"]}, {"bench_pa", "bench_pb", "bench_pc"})
        pair = next(p for p in res["pairs"] if {p["a"], p["b"]} == {"pa", "pc"})
        self.assertEqual(pair["shared"], ["a.py"])
        self.assertEqual(pair["conflicts"], ["a.py"])
        apart = next(p for p in res["pairs"] if {p["a"], p["b"]} == {"pa", "pb"})
        self.assertEqual((apart["shared"], apart["conflicts"]), ([], []))
        self.assertIn("spawn_agent", res["next_step"])
        self.assertIn("ask_user_question", res["next_step"])
        # One after the other in one tree: nothing to warn of.
        self.assertFalse(any(p["together"] for p in res["pairs"]))
        self.assertEqual(res["warnings"], [])

    def test_a_proxy_with_nothing_kept_is_set_aside(self) -> None:
        server_proxy.proxy_manage(op="register", name="pd", executable_path=self.harness,
                                  run_cmd_template="python3 {executable}", confirm=True)
        _eval(op="init", proxy_name="pd", benchmark_name="bench_pa",
              requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
              proxy_source_path=self.harness, optimize_paths=[os.path.join(self.root, "c.py")], confirm=True)
        res = server_proxy.proxy_merge(op="plan", proxies=["pa", "pd"])
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["refused"][0]["proxy"], "pd")

    def test_a_patch_holds_only_its_proxys_own_paths(self) -> None:
        """A snapshot taken while copies shared the store's index held the other proxies'
        staged files too: the patch keeps those under the proxy's optimize_paths."""
        cfg = {"root": "/w", "optimize_paths": ["/w/src/iso_dir.f90", "/w/lib"]}
        files = ["lib/m.f90", "lib2/x.f90", "src/iso_adj.f90", "src/iso_dir.f90"]
        self.assertEqual(merge_sources.own(cfg, files), ["lib/m.f90", "src/iso_dir.f90"])

    def test_a_frozen_session_is_left_out_and_says_why(self) -> None:
        res = server_proxy.proxy_merge(op="plan", proxies=["pa", "pb", "pa@20261104-101500"])
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["refused"], [{"proxy": "pa@20261104-101500",
                                           "why": "a frozen session: its trees are no longer kept"}])

    def test_start_is_refused_in_the_main_workspace(self) -> None:
        res = server_proxy.proxy_merge(op="start", proxies=["*"], confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("plan", res["hint"])


class KeptCommitTests(_MergeFixture):
    def test_the_kept_state_is_the_best_run_not_a_rejected_attempt_and_no_branch(self) -> None:
        self._write({"a.py": "A = 9\n\n\n\n\nX = 0\n"})
        self.assertEqual(self._settle("pa", 12.0)["verdict"], "reject")
        self._write(ORIGINAL)
        branches = self._git("branch", "--list")
        src, _ = merge_sources.source("pa")
        sha, why = merge_sources.kept_commit(src)
        self.assertEqual(why, "")
        self.assertEqual(self._git("show", f"{sha}:a.py"), "A = 2\n\n\n\n\nX = 0\n")
        # Its parent is pa's baseline, so a pick is a three-way merge from there.
        self.assertEqual(self._git("show", f"{sha}^:a.py"), ORIGINAL["a.py"])
        self.assertEqual(self._git("branch", "--list"), branches)
        self.assertEqual(self._read("a.py"), ORIGINAL["a.py"])     # nothing checked out


    def test_a_proxy_run_after_another_in_this_session_picks_only_its_own_work(self) -> None:
        """pe starts from pa's kept code, in the same tree, and changes another line of
        a.py: merged after pa, only pe's line is picked, without a conflict."""
        self._write({"a.py": "A = 2\n\n\n\n\nX = 0\n"})
        self._source("pe", {"a.py": "A = 2\n\n\n\n\nX = 1\n"}, 9.5)
        self._write(ORIGINAL)
        for proxy in ("pa", "pe"):
            src, _ = merge_sources.source(proxy)
            sha, _ = merge_sources.kept_commit(src)
            self.assertEqual(merge_sources.git("cherry-pick", sha).returncode, 0, proxy)
        self.assertEqual(self._read("a.py"), "A = 2\n\n\n\n\nX = 1\n")


class MergeRunTests(_MergeFixture):
    def test_a_conflict_stops_the_merge_with_both_sides(self) -> None:
        res = self._merge()
        c = res["conflict"]
        self.assertEqual(c["proxy"], "pc")
        self.assertEqual([f["path"] for f in c["files"]], ["a.py"])
        hunk = c["files"][0]["hunks"][0]
        self.assertIn("A = 2", hunk)
        self.assertIn("A = 3", hunk)
        self.assertEqual(c["ours"]["proxies"], ["pa"])
        self.assertIn("resolved", res["next_step"])
        self.assertIn("ask_user_question", res["note"])

    def test_resolved_is_refused_while_a_marker_is_left(self) -> None:
        self._merge()
        res = server_proxy.proxy_merge(op="resolved", resolution="kept both", confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("a.py", res["error"])

    def test_resolved_goes_on_to_the_next_source_and_the_measure(self) -> None:
        self._merge()
        res = self._resolve_keeping_pa()
        self.assertIn("merge: pa + pc + pb", res["next_step"])
        self.assertEqual((self._read("a.py"), self._read("b.py")), ("A = 2\n\n\n\n\nX = 0\n", "B = 2\n"))
        self.assertEqual(self._git("status", "--porcelain", "a.py", "b.py"), "")

    def test_each_kernel_against_its_gain_alone_and_nothing_set_aside(self) -> None:
        self._merge()
        self._resolve_keeping_pa()
        self._complete("merge-t", {"pa/a": 8.0, "pc/a": 10.0, "pb/a": 9.0})
        rep = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(rep["phase"], "done")
        k = {x["proxy"]: x for x in rep["kernels"]}
        self.assertAlmostEqual(k["pa"]["in_merge"], 0.8, places=5)
        self.assertAlmostEqual(k["pa"]["interaction"], 1.0, places=5)
        self.assertTrue(k["pc"]["flagged"])                 # its rewrite lost to pa's
        self.assertFalse(k["pb"]["flagged"])
        self.assertAlmostEqual(rep["whole"]["relative"], 0.9, places=5)
        self.assertEqual(rep["dropped"], [])
        self.assertIn("pc lost gain", rep["next_step"])
        merged = {m["proxy"]: m for m in rep["merged"]}
        self.assertEqual(merged["pc"]["conflicts"], ["a.py"])
        self.assertIn("method choice", merged["pc"]["resolution"])

    def test_drop_takes_a_source_back_out_and_measures_again(self) -> None:
        self._merge()
        self._resolve_keeping_pa()
        self._complete("merge-t", {"pa/a": 8.0, "pc/a": 10.0, "pb/a": 9.0})
        server_proxy.proxy_merge(op="next", confirm=True)
        res = server_proxy.proxy_merge(op="drop", proxies=["pb"], confirm=True)
        self.assertIn("merge: pa + pc", res["next_step"])
        self.assertEqual(self._read("b.py"), "B = 1\n")
        self._complete("merge-t", {"pa/a": 8.0, "pc/a": 10.0, "pb/a": 10.0})
        rep = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(rep["dropped"], ["pb"])
        self.assertNotIn("pb", {x["proxy"] for x in rep["kernels"]})

    def test_pause_undoes_the_pending_pick_and_next_carries_on(self) -> None:
        self._merge()
        res = server_proxy.proxy_merge(op="pause", question="pa's tiling or pc's?", confirm=True)
        self.assertEqual(res["question"]["proxy"], "pc")
        self.assertEqual(self._read("a.py"), "A = 2\n\n\n\n\nX = 0\n")
        self.assertEqual(self._git("status", "--porcelain", "a.py"), "")
        again = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(again["conflict"]["proxy"], "pc")

    def test_work_already_committed_on_the_branch_is_measured_out_then_merged_back(self) -> None:
        """pa's work committed on this branch, pc's not: the copy untouched is measured
        without pa, so pa's kernel keeps its gain instead of seeming to lose it."""
        self._write({"a.py": "A = 2\n\n\n\n\nX = 0\n"})
        self._git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "pa's work")
        with mock.patch.object(merge, "_in_copy", return_value=True):
            res = server_proxy.proxy_merge(op="start", proxies=["*"], proxy_name="merge-t", confirm=True)
        self.assertEqual(res["unapplied"], ["pa"])
        self.assertEqual(self._read("a.py"), ORIGINAL["a.py"])
        server_proxy.proxy_merge(op="next", confirm=True)
        self._complete("merge-t", {"pa/a": 10.0, "pc/a": 10.0, "pb/a": 10.0})
        conflict = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(conflict["conflict"]["proxy"], "pc")
        self._resolve_keeping_pa()
        self._complete("merge-t", {"pa/a": 8.0, "pc/a": 10.0, "pb/a": 9.0})
        rep = server_proxy.proxy_merge(op="next", confirm=True)
        pa = next(k for k in rep["kernels"] if k["proxy"] == "pa")
        self.assertAlmostEqual(pa["in_merge"], 0.8, places=5)
        self.assertFalse(pa["flagged"])

    def test_a_pick_this_copy_already_holds_is_not_a_failure(self) -> None:
        src, _ = merge_sources.source("pb")
        sha, _ = merge_sources.kept_commit(src)
        self.assertEqual(merge_sources.git("cherry-pick", sha).returncode, 0)
        again = merge_sources.git("cherry-pick", "--keep-redundant-commits", sha)
        self.assertEqual(again.returncode, 0, again.stderr)

    def test_next_waits_for_the_run_it_named(self) -> None:
        self._merge()
        res = self._resolve_keeping_pa()
        again = server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(again["next_step"], res["next_step"])

    def test_a_merge_leaves_only_its_branch_and_clean_takes_the_rest(self) -> None:
        branches = self._git("branch", "--list")
        self._merge()
        self._resolve_keeping_pa()
        self._complete("merge-t", {"pa/a": 8.0, "pc/a": 10.0, "pb/a": 9.0})
        server_proxy.proxy_merge(op="next", confirm=True)
        self.assertEqual(self._git("branch", "--list"), branches)
        reg = store.load_registry()
        self.assertIn("merge-t--pa", reg)
        server_proxy.proxy_manage(op="clean", name="merge-t", confirm=True)
        reg = store.load_registry()
        self.assertFalse([e for e in reg if e.startswith("merge-t--")])
        self.assertFalse(os.path.exists(store.suite_path("merge-t")))

    def test_the_map_draws_the_merge_and_keeps_it_frozen(self) -> None:
        self._merge()
        self._resolve_keeping_pa()
        self._complete("merge-t", {"pa/a": 8.0, "pc/a": 10.0, "pb/a": 9.0})
        server_proxy.proxy_merge(op="next", confirm=True)
        drawn = tab.answer("merge-t")
        self.assertEqual(drawn["merge"]["phase"], "done")
        meta = {m["name"]: m for m in drawn["proxies_meta"]}
        self.assertTrue(meta["merge-t"]["merge"])
        self.assertEqual(meta["merge-t"]["sources"], ["pa", "pc", "pb"])
        self.assertNotIn("merge-t--pa", meta)
        key = session.freeze("merge-t", "cleaned")
        self.assertEqual(session.session(key)["merge"]["phase"], "done")

    def test_ticked_proxies_bring_their_forecast(self) -> None:
        drawn = tab.answer(selected=["pa", "pb"])
        self.assertTrue(drawn["combined"])
        self.assertEqual(drawn["selected"], ["pa", "pb"])
        self.assertEqual([s["proxy"] for s in drawn["forecast"]["sources"]], ["pa", "pb"])
        self.assertEqual(drawn["forecast"]["refusal"], "")


class TogetherTests(unittest.TestCase):
    """Two proxies that ran at the same time in one tree, on a file both change."""

    def _src(self, root: str, start: str, end: str) -> dict:
        return {"root": root, "span": [f"2026-11-05T{start}:00+00:00", f"2026-11-05T{end}:00+00:00"]}

    def test_only_the_same_tree_at_the_same_time(self) -> None:
        a = self._src("/w", "10:00", "12:00")
        self.assertTrue(merge_sources.together(a, self._src("/w", "11:00", "13:00")))
        self.assertFalse(merge_sources.together(a, self._src("/w", "12:30", "13:00")))     # one after the other
        self.assertFalse(merge_sources.together(a, self._src("/copy", "11:00", "13:00")))  # copies of their own
        self.assertFalse(merge_sources.together(a, {"root": "/w", "span": []}))

    def test_the_plan_warns_of_a_pair_that_ran_together(self) -> None:
        said = merge.warnings([{"a": "pa", "b": "pb", "shared": ["a.py"], "conflicts": [], "together": True},
                               {"a": "pa", "b": "pc", "shared": ["a.py"], "conflicts": ["a.py"], "together": False}])
        self.assertEqual(len(said), 1)
        self.assertIn("pa and pb ran at the same time in the same tree and both change a.py", said[0])


class LegacyMergeTests(unittest.TestCase):
    def test_a_merge_frozen_before_git_merges_is_drawn_as_done(self) -> None:
        old = {"phase": "done", "sources": [{"proxy": "pa"}, {"proxy": "pb"}], "pairs": [], "expected": 0.72,
               "all": {"applied": ["pa", "pb"]}, "steps": [],
               "final": {"from": "chain", "relative": 0.8, "kept": ["pa"],
                         "set_aside": [{"proxy": "pb", "why": "interference"}]}}
        view = tab.merge_view(old)
        self.assertEqual(view["phase"], "done")
        self.assertEqual([m["proxy"] for m in view["merged"]], ["pa"])
        self.assertEqual(view["dropped"], ["pb"])
        self.assertEqual(view["whole"], {"relative": 0.8})
        new = {"phase": "merging", "sources": [], "pairs": [], "merged": []}
        self.assertIs(tab.merge_view(new), new)


class MovedTests(unittest.TestCase):
    def test_every_path_under_the_root_is_moved_and_no_other(self) -> None:
        entry = {"executable_path": "/copies/a/tiny.py",
                 "run_cmd_template": "python3 /copies/a/run.py --in /copies/a/data /copies/ab/x",
                 "tags": ["/copies/a"], "build_cwd": "/copies/a", "peak": 1.0}
        self.assertEqual(merge._moved(entry, "/copies/a", "/copies/m"), {
            "executable_path": "/copies/m/tiny.py",
            "run_cmd_template": "python3 /copies/m/run.py --in /copies/m/data /copies/ab/x",
            "tags": ["/copies/m"], "build_cwd": "/copies/m", "peak": 1.0})

