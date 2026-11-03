"""R&D requests from the code map: numbered as they are opened, started from where the
session stood, their variants each restored to that start, and their runs found by axis."""

import os
import tempfile
import unittest
from unittest import mock

from mimir.servers.proxy._lib import layout, requests, tree_snapshot
from mimir.servers.proxy._ops import code_map_steps, eval_session, requests_ops


class _Store(unittest.TestCase):
    """Each test's proxy_bench in a directory of its own."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = layout._ROOT
        layout._ROOT = os.path.join(self.tmp.name, "proxy_bench")
        self.addCleanup(setattr, layout, "_ROOT", saved)


class StoreTests(_Store):
    def test_requests_are_numbered_in_order(self) -> None:
        self.assertEqual([requests.add({"idea": i})["id"] for i in "ab"], ["R1", "R2"])
        self.assertEqual(requests.get("2")["idea"], "b")
        self.assertEqual(requests.update("r1", {"status": "done"})["status"], "done")
        self.assertIsNone(requests.update("R9", {"status": "done"}))

    def test_an_axis_names_its_request_and_variant(self) -> None:
        self.assertEqual(requests.axis_of("[R3] multigrid"), ("R3", None))
        self.assertEqual(requests.axis_of(" [R10.2] BiCGStab"), ("R10", 2))
        self.assertIsNone(requests.axis_of("tiling [R3]"))


class RefactorTests(_Store):
    def test_a_refactor_run_is_kept_when_neutral_and_only_a_refactor(self) -> None:
        from mimir.servers.proxy._lib import ratchet
        best = {"primary_value": 10.0}
        self.assertEqual(ratchet.ratchet_verdict(True, 10.1, best, "min", 0.02, neutral=True), "accept")
        self.assertEqual(ratchet.ratchet_verdict(True, 10.5, best, "min", 0.02, neutral=True), "reject")
        self.assertEqual(ratchet.ratchet_verdict(True, 10.1, best, "min", 0.02), "reject")
        self.assertEqual(ratchet.ratchet_verdict(True, 9.9, best, "max", 0.02, neutral=True), "accept")
        self.assertEqual(ratchet.ratchet_verdict(False, 9.0, best, "min", 0.02, neutral=True), "reject")

    def test_only_a_refactor_request_s_axis_is_neutral(self) -> None:
        requests.add({"kind": "method"})
        requests.add({"kind": "refactor"})
        self.assertFalse(requests.is_refactor("[R1] multigrid"))
        self.assertTrue(requests.is_refactor("[R2.1] break the cycle"))
        self.assertFalse(requests.is_refactor("break the cycle"))

    def test_the_whole_code_and_a_component_are_targets(self) -> None:
        with mock.patch.object(requests_ops, "start_of", return_value={}):
            row = requests_ops.open_("", "t", "study", "refactor", [{"all": True}, {"component": "solver"}],
                                     "decouple", [], "", "")["request"]
        self.assertEqual(row["targets"], [{"all": True}, {"component": "solver"}])


class OpenTests(_Store):
    def _open(self, mode: str = "try", **start) -> dict:
        with mock.patch.object(requests_ops, "start_of", return_value=start), \
             mock.patch.object(requests_ops, "resolve_proxy_name", return_value="solver"):
            return requests_ops.open_("", "tok1", mode, "method",
                                      [{"file": "pkg/a.py", "function": "relax", "line": 12, "extra": 1}, "pkg/b.py"],
                                      "multigrid instead of Jacobi", ["V-cycle", " ", "W-cycle"], "", "")

    def test_a_request_starts_where_the_session_stands(self) -> None:
        out = self._open(proxy="solver", run_id="r4", tree="t4", value=8.0)
        row = out["request"]
        self.assertEqual((row["id"], row["status"], row["start"]["tree"]), ("R1", "open", "t4"))
        self.assertEqual(row["targets"], [{"file": "pkg/a.py", "function": "relax", "line": 12}, {"file": "pkg/b.py"}])
        self.assertEqual(row["variants"], ["V-cycle", "W-cycle"])
        self.assertIn("reset_to_request", out["next_step"])
        self.assertIn("[R1.k]", out["next_step"])

    def test_the_same_message_opens_one_request(self) -> None:
        self._open(proxy="solver", run_id="r4", tree="t4", value=8.0)
        again = self._open(proxy="solver", run_id="r4", tree="t4", value=8.0)
        self.assertEqual(again["request"]["id"], "R1")
        self.assertEqual(len(requests.load()), 1)

    def test_a_study_edits_nothing_and_a_try_without_session_sets_one_up(self) -> None:
        self.assertIn("Edit no code", self._open("study")["next_step"])
        self.assertIn("proxy-optimize", requests_ops.open_("", "tok2", "try", "method", [], "x", [], "", "")["next_step"])

    def test_a_bad_mode_kind_or_status_is_refused(self) -> None:
        self.assertIn("error", requests_ops.open_("", "", "build", "method", [], "x", [], "", ""))
        self.assertIn("error", requests_ops.open_("", "", "try", "magic", [], "x", [], "", ""))
        self.assertIn("error", requests_ops.update("R1", "finished", "", ""))


class FollowUpTests(_Store):
    """A request continued from the chat rather than from the map still shows its runs."""

    def test_a_request_remembers_the_conversation_it_was_put_in(self) -> None:
        with mock.patch.object(requests_ops, "start_of", return_value={}), \
             mock.patch.object(requests_ops, "active_session_id", return_value="s-42"):
            row = requests_ops.open_("", "t", "study", "method", [{"file": "a.py"}], "x", [], "", "")["request"]
        self.assertEqual(row["session"], "s-42")

    def test_a_run_without_the_tag_is_attached_after_the_fact(self) -> None:
        requests.add({"mode": "study", "status": "proposed", "idea": "x",
                      "start": {"proxy": "solver", "value": 10.0}})
        self.assertEqual(requests_ops.update("R1", "trying", "", "", run_id="r9")["request"]["runs"], ["r9"])
        self.assertEqual(requests_ops.update("R1", "", "", "", run_id="r9")["request"]["runs"], ["r9"])
        steps = [{"run_id": "r9", "axis": "multigrid", "verdict": "accept", "value": 8.0}]
        with mock.patch.object(code_map_steps, "session", return_value={"steps": steps}):
            (row,) = code_map_steps.requests_view()
        self.assertEqual([(s["run_id"], s["variant"]) for s in row["steps"]], [("r9", None)])
        self.assertEqual(row["status"], "trying")

    def test_an_untagged_run_is_reminded_of_the_open_requests(self) -> None:
        requests.add({"status": "proposed", "idea": "multigrid", "start": {"proxy": "solver"}})
        requests.add({"status": "done", "idea": "old", "start": {"proxy": "solver"}})
        requests.add({"status": "open", "idea": "elsewhere", "start": {"proxy": "mesh"}})
        hint = requests_ops.untagged_hint("solver", "multigrid V-cycle", "r9")
        self.assertEqual([r["id"] for r in hint["open_requests"]], ["R1"])
        self.assertIn("run_id='r9'", hint["requests_note"])
        self.assertEqual(requests_ops.untagged_hint("solver", "[R1.1] V-cycle", "r9"), {})
        self.assertEqual(requests_ops.untagged_hint("mesh2", "x", "r9"), {})


class RestoreTests(_Store):
    def test_each_variant_starts_from_the_request_s_tree_not_the_best(self) -> None:
        root = os.path.join(self.tmp.name, "ws")
        git = os.path.join(self.tmp.name, "store", "opt.git")
        os.makedirs(os.path.join(root, "pkg"))
        os.makedirs(os.path.dirname(git))
        path = os.path.join(root, "pkg", "a.py")

        def write(text: str) -> None:
            with open(path, "w") as fh:
                fh.write(text)
        write("jacobi\n")
        start = tree_snapshot.snapshot(git, root, [path], "start")
        requests.add({"start": {"proxy": "solver", "run_id": "r4", "tree": start, "value": 8.0}})
        write("variant one, kept as the best\n")
        cfg = {"proxy_name": "solver", "optimize_paths": [path]}
        with mock.patch.object(eval_session, "load_opt_config", return_value=cfg), \
             mock.patch.object(eval_session, "opt_git_dir", return_value=git), \
             mock.patch.object(eval_session, "workspace_root", return_value=root):
            out = eval_session.reset_to_request("R1")
        self.assertEqual(out["from_run"], "r4")
        with open(path) as fh:
            self.assertEqual(fh.read(), "jacobi\n")
        self.assertIn("error", eval_session.reset_to_request("R7"))


class MapTests(_Store):
    def test_a_request_s_runs_are_found_by_axis_against_its_start(self) -> None:
        requests.add({"mode": "try", "status": "open", "idea": "solvers",
                      "start": {"proxy": "solver", "value": 10.0, "metric": "time_s"}})
        steps = [{"run_id": "r1", "axis": "[R1.1] CG", "verdict": "accept", "value": 8.0},
                 {"run_id": "r2", "axis": "[R1.2] BiCGStab", "verdict": "reject", "value": 9.0},
                 {"run_id": "r3", "axis": "[R10] other", "verdict": "accept", "value": 7.0},
                 {"run_id": "r4", "axis": "tiling", "verdict": "accept", "value": 7.5}]
        with mock.patch.object(code_map_steps, "session", return_value={"steps": steps}):
            (row,) = code_map_steps.requests_view()
        self.assertEqual([(s["run_id"], s["variant"]) for s in row["steps"]], [("r1", 1), ("r2", 2)])
        self.assertAlmostEqual(row["steps"][1]["delta"], -0.1)
        self.assertEqual((row["status"], row["start_value"]), ("trying", 10.0))


if __name__ == "__main__":
    unittest.main()
