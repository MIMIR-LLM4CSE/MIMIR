"""A session optimises several metrics under constraints: the verdict, the spec, a tradeoff
judged, a best taken back, and a best retired by a constraint tightened."""
from __future__ import annotations

import json
import os
import unittest

from mimir.servers.proxy import server_proxy
from mimir.servers.proxy._lib import layout, ratchet, spec as spec_lib, store
from mimir.servers.proxy._ops import _eval_ratchet
from mimir.servers.workspace import server_bash
from mimir.tests.support.proxy import _eval
from mimir.tests.test_proxy_opt_loop import _RatchetFixture

TIME_ERR = [{"metric": "time_s", "goal": "min"}, {"metric": "err", "goal": "min"}]


class VerdictTests(unittest.TestCase):
    def test_each_objective_bettered_worsened_or_held(self) -> None:
        best = {"time_s": 10.0, "err": 1e-6}
        margins = {"time_s": 0.02, "err": 0.0}
        tolerant = [TIME_ERR[0], {**TIME_ERR[1], "tolerance": 0.5}]
        cases = [
            ("faster, as exact", True, {"time_s": 9.0, "err": 1e-6}, TIME_ERR, False, "accept"),
            ("as fast, more exact", True, {"time_s": 10.0, "err": 5e-7}, TIME_ERR, False, "accept"),
            ("faster, less exact", True, {"time_s": 9.0, "err": 2e-6}, TIME_ERR, False, "tradeoff"),
            ("faster, less exact within tolerance", True, {"time_s": 9.0, "err": 1.4e-6}, tolerant, False, "accept"),
            ("no better on any", True, {"time_s": 10.1, "err": 1e-6}, TIME_ERR, False, "reject"),
            ("a refactor no worse", True, {"time_s": 10.1, "err": 1e-6}, TIME_ERR, True, "accept"),
            ("a refactor worse on one", True, {"time_s": 10.1, "err": 2e-6}, TIME_ERR, True, "reject"),
            ("infeasible", False, {"time_s": 1.0, "err": 0.0}, TIME_ERR, False, "reject"),
            ("one objective, as before", True, {"time_s": 9.0}, TIME_ERR[:1], False, "accept"),
        ]
        for label, feasible, values, objectives, neutral, expected in cases:
            with self.subTest(label):
                verdict, _better, _worse = ratchet.objectives_verdict(
                    feasible, values, best, objectives, margins, neutral=neutral)
                self.assertEqual(verdict, expected)

    def test_a_spec_is_refused_with_what_is_wrong(self) -> None:
        declared = {"declare": [{"metric": "err", "goal": "min"}]}
        cases = [
            ({}, "at least one objective"),
            ({"objectives": [{"metric": "speed"}]}, "no metric is named 'speed'"),
            ({"objectives": [{"metric": "time_s"}, {"metric": "time_s"}]}, "already an objective"),
            ({"objectives": [{"metric": "l2_rel"}]}, "needs a sealed reference"),
            ({"objectives": [{"metric": "time_s", "tolerance": -1}]}, "tolerance"),
            ({"objectives": [{"metric": "time_s"}], "constraints": [{"metric": "time_s", "operator": "lte"}]},
             "a threshold, of_baseline"),
            ({"objectives": [{"metric": "time_s"}], "directives": [""]}, "sentences"),
            ({"objectives": [{"metric": "time_s"}], "bound": 0.003}, "bound_basis"),
            ({**declared, "objectives": TIME_ERR, "directives": ["stay in Fortran"], "bound": 0.003,
              "bound_basis": "3 fields x 4 B x 512^3 / 2 TB/s",
              "constraints": [{"metric": "time_s", "operator": "lte", "of_baseline": 1.1}]}, None),
        ]
        for spec, refusal in cases:
            with self.subTest(refusal):
                why = spec_lib.check(spec, {}, {"cases": [{"case_id": "main"}]}, {})
                if refusal is None:
                    self.assertIsNone(why)
                else:
                    self.assertIn(refusal, why or "")


class SettleTests(_RatchetFixture):
    def setUp(self) -> None:
        super().setUp()
        self._init(max_stall=9)
        said = _eval(op="configure", confirm=True, spec={
            "objectives": TIME_ERR, "declare": [{"metric": "err", "goal": "min"}],
            "constraints": [{"metric": "time_s", "operator": "lt", "threshold": 100.0},
                            {"metric": "peak_rss_mb", "operator": "lte", "of_baseline": 2.0}],
            "directives": ["stay in C++"]})
        self.assertEqual(said.get("status"), "ok", said)

    def _settle(self, run_id: str, time_s: float, err: float, peak: float = 100.0, passed: bool = True) -> dict:
        self._write_source(run_id)
        run_dir = self._complete_run(run_id, all_passed=passed, time_s=time_s, extra={"err": err})
        metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
        metrics["peak_rss_mb"] = peak
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh)
        return self._results()

    def test_a_tradeoff_waits_for_a_judgement_and_a_relative_constraint_gates(self) -> None:
        cfg = store.load_opt_config("tiny")
        self.assertEqual((cfg["primary_metric"], cfg["secondary"][0]["metric"]), ("time_s", "err"))
        self.assertEqual(self._settle("r1", 10.0, 1e-6)["verdict"], "accept")
        traded = self._settle("r2", 8.0, 3e-6)
        self.assertEqual((traded["verdict"], traded["better"], traded["worse"]), ("tradeoff", ["time_s"], ["err"]))
        self.assertIn("A tradeoff: better on time_s, worse on err", traded["recommendation"])
        self.assertIn("report_verdict(run='r2'", traded["next_step"])
        self.assertEqual(ratchet.load_best("tiny")["run_id"], "r1")
        self.assertEqual(store.load_opt_config("tiny")["stall"], 0)

        # The agent's verdict on the run is its judgement in the ledger.
        said = server_bash.report_verdict(verdict="pass", run="r2", reason="20% faster, error far under tolerance")
        self.assertEqual((said["ratchet"]["verdict"], said["ratchet"]["judged_by"]), ("accept", "agent"), said)
        self.assertEqual(ratchet.load_best("tiny")["run_id"], "r2")

        # Twice the baseline's memory: broken, whatever it gains.
        broken = self._settle("r3", 5.0, 1e-7, peak=250.0)
        self.assertEqual((broken["verdict"], broken["broken"][0]["metric"]), ("reject", "peak_rss_mb"))
        self.assertEqual(self._settle("r4", 7.0, 2e-6)["verdict"], "accept")
        # The best taken back (it is wrong, or breaks a directive): the one before it returns.
        back = server_bash.report_verdict(verdict="rejected", run="r4", reason="rewrote it in Fortran")["ratchet"]
        self.assertEqual(back["best"]["run_id"], "r2")
        self.assertIn("reset_to_best", back["note"])
        # A run neither a tradeoff nor the best: the verdict leaves the ratchet alone.
        self.assertNotIn("ratchet", server_bash.report_verdict(verdict="pass", run="r3", reason="x"))

    def test_a_tightened_constraint_retires_a_best_that_fails_it(self) -> None:
        """A best fast because it was wrong is no value to beat: once the constraint that
        catches it is in the spec, the last kept run that meets it is the best again."""
        self.assertEqual(self._settle("r1", 10.0, 1e-7)["verdict"], "accept")
        self.assertEqual(self._settle("r2", 8.0, 1e-7)["verdict"], "accept")
        self.assertEqual(self._settle("r3", 6.0, 1e-7)["verdict"], "accept")
        # r2 and r3 were wrong at the edge: their error says so once it is bounded.
        for run_id in ("r2", "r3"):
            path = os.path.join(layout.opt_session_runs_dir("tiny"), run_id, "metrics.json")
            metrics = json.load(open(path))
            metrics["results"][0]["metrics"]["err"] = 1e-3
            json.dump(metrics, open(path, "w"))
        tighten = {"constraints": [{"metric": "time_s", "operator": "lt", "threshold": 100.0},
                                   {"metric": "err", "operator": "lte", "threshold": 1e-6}]}
        refused = _eval(op="configure", spec=tighten, confirm=True)
        self.assertIn("reason", refused["error"])
        said = _eval(op="configure", spec={**tighten, "reason": "the boundary is fixed"}, confirm=True)
        self.assertEqual(said.get("status"), "ok", said)
        self.assertEqual((said["retired"]["retired"], said["retired"]["also_retired"]), ("r3", ["r2"]))
        self.assertEqual(ratchet.load_best("tiny")["run_id"], "r1")
        self.assertIn("r1 is the best again", said["note"])
        rows = {r["run_id"]: r for r in ratchet.read_ledger("tiny")}
        self.assertEqual(rows["r3"]["retired"]["reason"], "the boundary is fixed")
        self.assertNotIn("retired", rows["r1"])
        # The corrected code, slower than the wrong best, is judged against what met the constraint.
        self.assertEqual(self._settle("r4", 7.0, 1e-7)["verdict"], "accept")
        history = server_proxy.proxy_eval_status(op="metrics")["history"]
        self.assertEqual(history[-1]["reason"], "the boundary is fixed")

    def test_a_measured_session_s_spec_changes_with_a_reason_kept_in_its_history(self) -> None:
        self._settle("r1", 10.0, 1e-6)
        change = {"change": [{"metric": "err", "tolerance": 0.5}]}
        refused = _eval(op="configure", spec=change, confirm=True)
        self.assertIn("reason", refused["error"])
        changed = _eval(op="configure", confirm=True, spec={**change, "reason": "a coarser error for speed"})
        self.assertEqual(changed["status"], "ok", changed)
        cfg = store.load_opt_config("tiny")
        self.assertEqual(cfg["secondary"], [{"metric": "err", "goal": "min", "tolerance": 0.5}])
        self.assertEqual(changed["loosened"], [{"metric": "err", "what": "tolerance 0 -> 0.5", "correctness": False}])
        listed = server_proxy.proxy_eval_status(op="metrics")
        self.assertEqual((listed["history"][0]["after_run"], listed["history"][0]["reason"]),
                         ("r1", "a coarser error for speed"))
        self.assertIn("err", {m["name"] for m in listed["metrics"]})
        self.assertIn("Nothing checks this session's solution", listed["warning"])
        # A correctness bound widened or dropped is told apart from a cost's.
        before = {"constraints": [{"metric": "l2_rel", "operator": "lte", "threshold": 1e-9},
                                  {"metric": "time_s", "operator": "lte", "of_baseline": 1.1}]}
        after = spec_lib.merged(before, {"change": [{"metric": "l2_rel", "threshold": 1e-6},
                                                    {"metric": "time_s", "remove": True}]})
        self.assertEqual([(x["metric"], x["correctness"]) for x in spec_lib.loosened(before, after, {})],
                         [("l2_rel", True), ("time_s", False)])


class PrecisionFloorTests(unittest.TestCase):
    def test_an_error_relative_to_the_baseline_may_reach_the_field_s_precision(self) -> None:
        """A reordered sum moves the last bits: bounded by the baseline's own spread alone,
        a float32 field could never be summed in another order."""
        req = {"metric": "l2_rel", "operator": "lte", "of_baseline": 10.0, "precision_floor": True}
        cfg = {"proxy_name": "p", "baseline_run_id": "b", "requirements": [req], "benchmark_name": "s"}
        cases = [("float32, reordered", "float32", 1e-6, []), ("float32, wrong", "float32", 1e-3, ["l2_rel"]),
                 ("float64, reordered", "float64", 1e-6, ["l2_rel"])]
        for label, dtype, err, broken in cases:
            with self.subTest(label):
                from unittest import mock
                with mock.patch.object(_eval_ratchet, "read_ledger",
                                       return_value=[{"run_id": "b", "values": {"l2_rel": 1e-12}}]), \
                     mock.patch.object(_eval_ratchet, "load_suite",
                                       return_value={"cases": [{"reference_name": "ref"}]}), \
                     mock.patch.object(_eval_ratchet, "load_ref_metrics", return_value={"dtype": dtype}):
                    found = _eval_ratchet._relative_broken(cfg, {"l2_rel": err})
                self.assertEqual([b["metric"] for b in found], broken)
