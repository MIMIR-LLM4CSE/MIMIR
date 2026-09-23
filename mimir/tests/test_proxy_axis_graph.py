"""The shape of an optimisation: which ideas were tried, and which are still open.

A session ledger records what each run measured. What it did not record is what the
run was *for* — so a session that had tried eight things was, on disk, eight numbers,
and the next agent to look had no way to tell an idea already ruled out from one nobody
had thought of. Delegating axes to parallel sub-agents multiplies that: each writes its
own ledger, and nothing joined them up.

These pin what the graph is allowed to claim. The edge is the incumbent a run was
measured against — not the tree it started from, which two runs share when they measure
the SAME code. And attempts made under one incumbent are independent of each other,
which is a reason to try combining them and never a promise that they add up.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROXY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "servers", "proxy")
sys.path.insert(0, os.path.abspath(_PROXY))
sys.path.insert(0, os.path.abspath(os.path.join(_PROXY, "..", "_shared")))

from _ops import eval_session  # noqa: E402


class _Ledgered(unittest.TestCase):
    """A store holding one or more hand-written ledgers."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.configs: dict[str, dict] = {}
        self.bests: dict[str, dict] = {}

    def _write(self, proxy: str, rows: list[dict], cfg: dict | None = None,
               best: dict | None = None, raw_tail: str = "") -> None:
        path = os.path.join(self.tmp, f"{proxy}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.write(raw_tail)
        self.configs[proxy] = cfg or {}
        self.bests[proxy] = best or {}

    def _graph(self, proxy_name: str = ""):
        with mock.patch.object(eval_session, "_load_registry",
                               return_value={k: {} for k in self.configs}), \
             mock.patch.object(eval_session, "_opt_ledger_file",
                               side_effect=lambda p: os.path.join(self.tmp, f"{p}.jsonl")), \
             mock.patch.object(eval_session, "_load_opt_config",
                               side_effect=lambda p: self.configs.get(p)), \
             mock.patch.object(eval_session, "_load_best",
                               side_effect=lambda p: self.bests.get(p)):
            return eval_session.graph(proxy_name)


class AxisTests(_Ledgered):
    def test_the_axis_travels_with_the_number_it_produced(self):
        """The one field of a run that cannot be measured."""
        self._write("solver", [
            {"run_id": "r1", "axis": "baseline", "verdict": "accept",
             "primary_value": 10.0, "feasible": True},
            {"run_id": "r2", "axis": "tiling", "verdict": "accept",
             "primary_value": 7.0, "feasible": True, "best_run_id": "r1"},
        ])
        axes = {n["run_id"]: n["axis"] for n in self._graph()["nodes"]}
        self.assertEqual(axes, {"r1": "baseline", "r2": "tiling"})

    def test_a_rejected_axis_is_kept_because_that_is_the_point(self):
        """An idea ruled out on this code is what stops the next run paying to learn
        the same thing again."""
        self._write("solver", [
            {"run_id": "r1", "axis": "baseline", "verdict": "accept"},
            {"run_id": "r2", "axis": "unrolling", "verdict": "reject",
             "best_run_id": "r1"},
        ])
        rejected = [n for n in self._graph()["nodes"] if n["verdict"] == "reject"]
        self.assertEqual([n["axis"] for n in rejected], ["unrolling"])


class LineageTests(_Ledgered):
    def test_the_edge_is_the_incumbent_the_run_was_measured_against(self):
        self._write("solver", [
            {"run_id": "r1", "verdict": "accept"},
            {"run_id": "r2", "verdict": "reject", "best_run_id": "r1"},
            {"run_id": "r3", "verdict": "accept", "best_run_id": "r1"},
        ])
        edges = [(e["from"], e["to"]) for e in self._graph()["edges"]]
        self.assertEqual(sorted(edges), [("r1", "r2"), ("r1", "r3")])

    def test_a_run_judged_against_itself_is_not_its_own_parent(self):
        """The first accepted run is its own best by the time it settles."""
        self._write("solver", [{"run_id": "r1", "verdict": "accept",
                                "best_run_id": "r1"}])
        self.assertEqual(self._graph()["edges"], [])

    def test_two_attempts_under_one_incumbent_are_independent_of_each_other(self):
        """Neither was made with the other in the tree, so neither contains the
        other's edit — which is what makes combining them worth a run."""
        self._write("solver", [
            {"run_id": "r1", "verdict": "accept", "axis": "baseline"},
            {"run_id": "r2", "verdict": "accept", "axis": "tiling",
             "primary_value": 7.0, "best_run_id": "r1"},
            {"run_id": "r3", "verdict": "accept", "axis": "vectorise",
             "primary_value": 8.0, "best_run_id": "r1"},
        ])
        same = self._graph()["from_the_same_point"]
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["incumbent"], "r1")
        self.assertEqual(sorted(a["axis"] for a in same[0]["attempts"]),
                         ["tiling", "vectorise"])

    def test_one_attempt_alone_is_not_a_pair(self):
        self._write("solver", [
            {"run_id": "r1", "verdict": "accept"},
            {"run_id": "r2", "verdict": "accept", "best_run_id": "r1"},
        ])
        self.assertEqual(self._graph()["from_the_same_point"], [])

    def test_nothing_in_the_answer_scores_how_well_two_axes_combine(self):
        """It is not derivable from what was measured. Only a run that combines them
        can answer it, and a number here would be read as if one had."""
        self._write("solver", [
            {"run_id": "r1", "verdict": "accept"},
            {"run_id": "r2", "verdict": "accept", "best_run_id": "r1"},
            {"run_id": "r3", "verdict": "accept", "best_run_id": "r1"},
        ])
        out = self._graph()
        keys = set(out["from_the_same_point"][0]) | {
            k for a in out["from_the_same_point"][0]["attempts"] for k in a}
        self.assertFalse({"score", "complementarity", "expected_gain"} & keys)


class SeveralAxesTests(_Ledgered):
    def test_the_sessions_of_a_fan_out_are_read_together(self):
        """Each sub-agent optimises a copy of its own and writes its own ledger; the
        picture is only worth anything whole."""
        self._write("axis_a", [{"run_id": "a1", "verdict": "accept",
                                "primary_value": 9.0}],
                    cfg={"branch": "mimir/sub-aaa", "primary_metric": "time_s"},
                    best={"run_id": "a1", "primary_value": 9.0})
        self._write("axis_b", [{"run_id": "b1", "verdict": "reject"}],
                    cfg={"branch": "mimir/sub-bbb", "primary_metric": "time_s"})
        out = self._graph()
        self.assertEqual({a["proxy"] for a in out["axes"]}, {"axis_a", "axis_b"})
        self.assertEqual({a["branch"] for a in out["axes"]},
                         {"mimir/sub-aaa", "mimir/sub-bbb"})
        self.assertEqual({a["proxy"] for a in out["axes"] if a["accepted"]}, {"axis_a"})

    def test_asking_for_one_session_reads_only_that_one(self):
        self._write("axis_a", [{"run_id": "a1", "verdict": "accept"}])
        self._write("axis_b", [{"run_id": "b1", "verdict": "accept"}])
        self.assertEqual([a["proxy"] for a in self._graph("axis_a")["axes"]], ["axis_a"])

    def test_two_sessions_never_share_an_edge(self):
        """Run ids are per session, so a shared id is a collision, not a lineage."""
        self._write("axis_a", [{"run_id": "r1", "verdict": "accept"},
                               {"run_id": "r2", "best_run_id": "r1"}])
        self._write("axis_b", [{"run_id": "r1", "verdict": "accept"},
                               {"run_id": "r3", "best_run_id": "r1"}])
        for edge in self._graph()["edges"]:
            self.assertIn(edge["proxy"], ("axis_a", "axis_b"))
        same = self._graph()["from_the_same_point"]
        # r1 of each session has one attempt under it — not two under a merged r1.
        self.assertEqual(same, [])


class OldAndHalfWrittenLedgersTests(_Ledgered):
    def test_entries_from_before_an_axis_was_recorded_still_read(self):
        """The ledger is append-only and outlives the format it started in."""
        self._write("solver", [{"run_id": "r1", "verdict": "accept",
                                "primary_value": 10.0}])
        node = self._graph()["nodes"][0]
        self.assertEqual(node["axis"], "")
        self.assertEqual(node["launch_tree"], "")
        self.assertEqual(node["primary_value"], 10.0)

    def test_a_half_written_last_line_costs_its_own_row_and_nothing_else(self):
        """It is written by another process, and the panel may read mid-append."""
        self._write("solver", [{"run_id": "r1", "verdict": "accept"}],
                    raw_tail='{"run_id": "r2", "verd')
        self.assertEqual([n["run_id"] for n in self._graph()["nodes"]], ["r1"])

    def test_a_session_with_no_ledger_at_all_is_simply_absent(self):
        self.configs["never_run"] = {"primary_metric": "time_s"}
        self.assertEqual(self._graph()["nodes"], [])


class AttachedToEveryRunTests(_Ledgered):
    """The account travels with the result, rather than waiting to be asked for.

    The moment the next edit is chosen is the moment a run's verdict comes back. A
    report the agent has to remember to request is one it will not request, and the
    cost of that is a full run spent re-learning something the ledger already knew.
    """

    def _axes(self, proxy: str = "solver"):
        with mock.patch.object(eval_session, "_resolve_proxy_name",
                               side_effect=lambda p: p or "solver"), \
             mock.patch.object(eval_session, "_opt_ledger_file",
                               side_effect=lambda p: os.path.join(self.tmp, f"{p}.jsonl")):
            return eval_session.axes_tried(proxy)

    def test_what_worked_and_what_was_ruled_out_both_come_back(self):
        self._write("solver", [
            {"run_id": "r1", "axis": "baseline", "verdict": "accept",
             "primary_value": 10.0},
            {"run_id": "r2", "axis": "unrolling", "verdict": "reject",
             "best_run_id": "r1"},
            {"run_id": "r3", "axis": "tiling", "verdict": "accept",
             "primary_value": 7.0, "best_run_id": "r1"},
        ])
        axes = self._axes()
        self.assertEqual(axes["tried"], 3)
        self.assertEqual(axes["rejected"], ["unrolling"])
        self.assertEqual({a["axis"] for a in axes["accepted"]}, {"baseline", "tiling"})

    def test_an_idea_rejected_twice_is_still_one_fact(self):
        """It is a list the model reads before choosing; the same word three times is
        three lines of context saying one thing."""
        self._write("solver", [
            {"run_id": f"r{i}", "axis": "unrolling", "verdict": "reject",
             "best_run_id": "r0"} for i in range(3)
        ])
        self.assertEqual(self._axes()["rejected"], ["unrolling"])

    def test_a_run_that_named_no_axis_is_counted_but_not_listed(self):
        """The count is measured; the name is the one thing only the model can supply,
        and an empty string in the ruled-out list would rule out nothing."""
        self._write("solver", [
            {"run_id": "r1", "verdict": "reject"},
            {"run_id": "r2", "axis": "tiling", "verdict": "reject"},
        ])
        axes = self._axes()
        self.assertEqual(axes["tried"], 2)
        self.assertEqual(axes["rejected"], ["tiling"])

    def test_it_says_what_the_lists_are_for(self):
        """Handed to a model with no note, a list of words is decoration."""
        self._write("solver", [{"run_id": "r1", "axis": "unrolling",
                                "verdict": "reject"}])
        self.assertIn("ruled out", self._axes()["note"])

    def test_a_session_with_no_runs_yet_attaches_nothing(self):
        """An empty structure on every early result is context spent saying nothing."""
        self._write("solver", [])
        self.assertEqual(self._axes(), {})

    def test_an_unreadable_ledger_costs_the_account_and_not_the_run(self):
        """A run's verdict must never fail over its own bookkeeping."""
        with mock.patch.object(eval_session, "_resolve_proxy_name",
                               side_effect=RuntimeError("store is gone")):
            self.assertEqual(eval_session.axes_tried("solver"), {})


class PanelLinesTests(unittest.TestCase):
    """The same account in the drawer, so the user sees it without asking either."""

    def _lines(self, axes: dict):
        import server_proxy
        with mock.patch.object(server_proxy.eval_session, "axes_tried",
                               return_value=axes):
            return server_proxy._axes_lines("solver")

    def test_the_shape_of_the_search_is_one_line(self):
        lines = self._lines({"tried": 7, "accepted": [{"axis": "tiling"}],
                             "rejected": ["unrolling", "simd"]})
        values = {line["label"]: line["value"] for line in lines}
        self.assertEqual(values["axes tried"], "7, 1 accepted")
        self.assertEqual(values["ruled out"], "unrolling, simd")

    def test_a_session_that_has_run_nothing_adds_no_lines(self):
        self.assertEqual(self._lines({}), [])

    def test_a_settled_run_shows_up_on_the_next_refresh(self):
        """The lines are cached on the ledger's own mtime, so there is no staleness to
        reason about: the moment a run settles, the next refresh re-reads it."""
        import server_proxy
        tmp = tempfile.mkdtemp()
        ledger = os.path.join(tmp, "ledger.jsonl")
        open(ledger, "w").close()
        answers = [{"tried": 1, "accepted": [], "rejected": ["unrolling"]},
                   {"tried": 2, "accepted": [], "rejected": ["unrolling", "simd"]}]

        with mock.patch("_lib.store._opt_ledger_file", return_value=ledger), \
             mock.patch.object(server_proxy.eval_session, "axes_tried",
                               side_effect=answers):
            first = server_proxy._axes_lines("solver")
            # Same mtime: served from the cache, so the second answer is not consumed.
            self.assertEqual(server_proxy._axes_lines("solver"), first)
            os.utime(ledger, (0, 0))    # a run settled
            second = server_proxy._axes_lines("solver")
        self.assertNotEqual(second, first)
        self.assertIn("simd", " ".join(line["value"] for line in second))

    def test_the_panel_survives_a_store_it_cannot_read(self):
        """One section that cannot answer costs its own lines, never the drawer."""
        import server_proxy
        with mock.patch.object(server_proxy.eval_session, "axes_tried",
                               side_effect=RuntimeError("store is gone")):
            self.assertEqual(server_proxy._axes_lines("solver"), [])


class RecordedAxisTests(unittest.TestCase):
    """What the launch wrote down, read back when the run settles hours later."""

    def test_the_axis_is_read_back_from_the_runs_own_config(self):
        with tempfile.TemporaryDirectory() as run_dir:
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"axis": "cache-friendly layout"}, fh)
            self.assertEqual(eval_session._run_axis(run_dir), "cache-friendly layout")

    def test_a_run_launched_without_one_settles_without_one(self):
        with tempfile.TemporaryDirectory() as run_dir:
            self.assertEqual(eval_session._run_axis(run_dir), "")


if __name__ == "__main__":
    unittest.main()
