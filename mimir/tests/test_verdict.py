"""The verdict line: the one thing in the ledger the model wrote.

Mimir never parses a program's output for a pass/fail — output is unbounded and belongs
to whoever wrote the code. It parses the model's *statement* about that output, which is
a format the model controls, and that asymmetry is what these tests pin: the grammar is
generous about the shapes a model writes and strict about not firing on prose.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mimir.client.context.execution_context import build_execution_context
from mimir.client.guardrails.verdict import parse_verdict, record_verdict


class ParseVerdictTests(unittest.TestCase):
    def test_the_three_outcomes_are_recognised(self) -> None:
        for word in ("pass", "fail", "unknown"):
            self.assertEqual(parse_verdict(f"verdict: {word} — because")[0], word)

    def test_separators_a_model_actually_writes(self) -> None:
        for line in (
            "verdict: pass — em dash",
            "verdict: pass - hyphen",
            "verdict: pass – en dash",
            "verdict = pass — equals",
            "- **verdict**: pass — bullet and bold",
            "> verdict: pass — quoted",
            "VERDICT: PASS — shouting",
        ):
            self.assertIsNotNone(parse_verdict(line), line)

    def test_the_reason_is_captured(self) -> None:
        _, reason, _ = parse_verdict("verdict: fail — l2_rel=0.4, well above the 1e-3 bound")
        self.assertEqual(reason, "l2_rel=0.4, well above the 1e-3 bound")

    def test_a_bare_verdict_still_parses(self) -> None:
        self.assertEqual(parse_verdict("verdict: unknown"), ("unknown", "", ""))

    def test_the_scope_bracket_is_captured(self) -> None:
        self.assertEqual(
            parse_verdict("verdict[python bench.py]: pass — 3e-4 against the analytic solution"),
            ("pass", "3e-4 against the analytic solution", "python bench.py"),
        )
        # Optional, and the unscoped form is untouched by the bracket in the grammar.
        self.assertEqual(parse_verdict("verdict: pass — fine")[2], "")

    def test_prose_is_not_a_verdict(self) -> None:
        # The word appears constantly in ordinary explanation; only the line form counts.
        for line in (
            "the verdict is still unclear",
            "I will give a verdict: after re-running the suite",
            "my verdict on this design: it passes for now",
        ):
            self.assertIsNone(parse_verdict(line), line)

    def test_the_last_statement_wins(self) -> None:
        # A model that reasons its way to a different conclusion meant the later one.
        text = "verdict: pass — looked fine\n…on reflection…\nverdict: fail — it is not"
        self.assertEqual(parse_verdict(text), ("fail", "it is not", ""))

    def test_empty_input_is_no_verdict(self) -> None:
        self.assertIsNone(parse_verdict(""))


class RecordVerdictTests(unittest.TestCase):
    def _ctx(self, command="pytest -q foo.py", paths=("foo.py",), tier="executed"):
        ec = build_execution_context()
        ec["dirty_written_files"] = set(paths)
        ec["code_mutation_started"] = True
        ec["unjudged_runs"] = {command: {"paths": list(paths), "tier": tier}}
        return ec

    def test_nothing_awaiting_means_nothing_recorded(self) -> None:
        ec = build_execution_context()
        self.assertFalse(record_verdict("verdict: pass — fine", ec))

    def test_a_pass_validates_at_the_runs_own_tier(self) -> None:
        ec = self._ctx(tier="oracle")
        self.assertTrue(record_verdict("verdict: pass — matches the reference", ec))
        self.assertIn("foo.py", ec["validated_files"])
        self.assertEqual(ec["validation_tier_by_file"]["foo.py"], "oracle")

    def test_one_statement_settles_runs_bearing_on_the_same_files(self) -> None:
        # The model saw both outputs and spoke after them, and there is nothing to
        # disambiguate: whichever run it meant, it meant that file.
        ec = self._ctx()
        ec["unjudged_runs"]["python foo.py"] = {"paths": ["foo.py"], "tier": "executed"}
        record_verdict("verdict: pass — within tolerance", ec)
        self.assertEqual(ec["validated_files"], {"foo.py"})
        self.assertFalse(ec["unjudged_runs"])

    def _two_runs(self):
        ec = self._ctx()
        ec["unjudged_runs"]["python bench.py"] = {"paths": ["bar.py"], "tier": "executed"}
        ec["dirty_written_files"].add("bar.py")
        return ec

    def test_an_unscoped_pass_over_different_files_credits_nothing(self) -> None:
        # It would credit a run the statement may never have been about — the same
        # over-crediting the verdict rule exists to stop, one level up.
        ec = self._two_runs()
        self.assertFalse(record_verdict("verdict: pass — both are within tolerance", ec))
        self.assertEqual(ec["validated_files"], set())
        self.assertEqual(len(ec["unjudged_runs"]), 2)
        self.assertEqual(
            sorted(ec["verdict_scope_required"]), ["pytest -q foo.py", "python bench.py"],
        )

    def test_a_scoped_pass_settles_only_the_run_it_names(self) -> None:
        ec = self._two_runs()
        self.assertTrue(record_verdict("verdict[bench.py]: pass — matches the reference", ec))
        self.assertEqual(ec["validated_files"], {"bar.py"})
        self.assertEqual(list(ec["unjudged_runs"]), ["pytest -q foo.py"])
        self.assertFalse(ec["verdict_scope_required"])

    def test_a_scope_matches_a_path_as_well_as_a_command(self) -> None:
        ec = self._two_runs()
        record_verdict("verdict[foo.py]: pass — the suite is green and correct", ec)
        self.assertEqual(ec["validated_files"], {"foo.py"})

    def test_a_scope_naming_nothing_outstanding_settles_nothing(self) -> None:
        ec = self._two_runs()
        self.assertFalse(record_verdict("verdict[unrelated.py]: pass — fine", ec))
        self.assertEqual(len(ec["unjudged_runs"]), 2)

    def test_a_path_less_probe_does_not_force_a_scope(self) -> None:
        # One probe and one real run is the commonest shape there is; the probe credits
        # nothing, so settling it alongside cannot over-credit anything.
        ec = self._ctx()
        ec["unjudged_runs"]["python /tmp/scratch/probe.py"] = {"paths": [], "tier": "executed"}
        self.assertTrue(record_verdict("verdict: pass — residual 3e-4", ec))
        self.assertEqual(ec["validated_files"], {"foo.py"})

    def test_fail_and_unknown_need_no_scope(self) -> None:
        # Withholding credit broadly is never the unsafe direction.
        for verdict in ("fail", "unknown"):
            ec = self._two_runs()
            self.assertTrue(record_verdict(f"verdict: {verdict} — off by a factor of two", ec))
            self.assertFalse(ec["unjudged_runs"])
            self.assertEqual(ec["validated_files"], set())

    def test_an_unknown_verdict_leaves_the_file_pending(self) -> None:
        ec = self._ctx()
        record_verdict("verdict: unknown — no reference for this regime", ec)
        self.assertNotIn("foo.py", ec["validated_files"])
        self.assertEqual(ec["verdict_by_file"]["foo.py"]["verdict"], "unknown")

    def test_a_resolved_run_re_arms_the_reminder_budget(self) -> None:
        ec = self._ctx()
        ec["nudge_counts"]["output_verdict"] = 2
        record_verdict("verdict: pass — fine", ec)
        self.assertEqual(ec["nudge_counts"]["output_verdict"], 0)

    def test_an_unknown_verdict_does_not_re_arm_it(self) -> None:
        # The condition it was spent on is still open; re-arming would be spam.
        ec = self._ctx()
        ec["nudge_counts"]["output_verdict"] = 2
        record_verdict("verdict: unknown — cannot tell", ec)
        self.assertEqual(ec["nudge_counts"]["output_verdict"], 2)


if __name__ == "__main__":
    unittest.main()
