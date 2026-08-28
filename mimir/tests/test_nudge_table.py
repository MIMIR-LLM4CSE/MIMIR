import unittest

from mimir.client.config.constants import EXERCISE_BUDGET
from mimir.client.guardrails.nudges.engine import (
    _ALL_GUIDANCE,
    _CORE_NUDGES,
    _CoreNudge,
    _guidance_enabled,
)
from mimir.client.guardrails.nudges.plugins import VALID_NUDGE_LAYERS


class CoreNudgeTableTest(unittest.TestCase):
    def test_all_rows_are_core_nudges(self):
        self.assertTrue(all(isinstance(n, _CoreNudge) for n in _CORE_NUDGES))

    def test_names_unique(self):
        names = [n.name for n in _CORE_NUDGES]
        self.assertEqual(len(names), len(set(names)))

    def test_layers_valid(self):
        for n in _CORE_NUDGES:
            self.assertIn(n.layer, VALID_NUDGE_LAYERS)

    def test_verification_before_guidance(self):
        # Ordering IS priority; verification rows must all precede guidance rows.
        layers = [n.layer for n in _CORE_NUDGES]
        first_guidance = layers.index("guidance")
        self.assertNotIn("verification", layers[first_guidance:])

    def test_verification_categories(self):
        verif = {n.name for n in _CORE_NUDGES if n.layer == "verification"}
        self.assertEqual(
            verif, {"denial", "error_recovery", "validation", "regression",
                    "unexercised", "unfinished_plan"}
        )

    def test_no_row_asks_for_a_verdict(self):
        # A verdict is asked for in-band (the VERDICT_DUE annotation on the run's own
        # result) and reported by the ledger when it never came. As a turn-end row it
        # fired on every green run, throwing away a finished answer for a label.
        self.assertNotIn("output_verdict", {n.name for n in _CORE_NUDGES})

    def test_the_exercise_rows_share_one_budget(self):
        # "run the existing test" and "nothing was exercised" are one question; two
        # budgets made it two re-prompts.
        shared = {n.name for n in _CORE_NUDGES if n.budget_key == EXERCISE_BUDGET}
        self.assertEqual(shared, {"regression", "unexercised"})

    def test_validation_is_the_only_required_row_of_the_three(self):
        # It precedes them, and it is the one nothing lets go of: a file the model
        # modified and never checked blocks the conclusion, an unrun one does not.
        names = [n.name for n in _CORE_NUDGES]
        for advisory in ("regression", "unexercised"):
            self.assertLess(names.index("validation"), names.index(advisory))
        self.assertEqual(
            _CORE_NUDGES[names.index("validation")].budget_key, "",
        )

    def test_verification_layer_is_never_enforcement_gated(self):
        # Verification rows check reality (disk/process state), not the model's
        # reasoning, so none of them may leak into the guidance table — that table
        # is the only thing enforcement can switch off. A verification nudge that
        # appeared in _ALL_GUIDANCE would silently go dead at enforcement="off".
        verif = {n.name for n in _CORE_NUDGES if n.layer == "verification"}
        self.assertEqual(verif & set(_ALL_GUIDANCE), set())

    def test_guidance_categories_match_guidance_table(self):
        # Every guidance category gated by _GUIDANCE_BY_LEVEL_MODE has exactly one row.
        guidance = {n.name for n in _CORE_NUDGES if n.layer == "guidance"}
        self.assertEqual(guidance, set(_ALL_GUIDANCE))

    def test_validation_is_not_enforcement_gated(self):
        # Checking a file one just modified is the working order's one requirement,
        # not a reasoning shim to dial down: it must survive enforcement="off".
        self.assertNotIn("validation", _ALL_GUIDANCE)
        for level in ("strict", "light", "off"):
            self.assertFalse(
                _guidance_enabled("validation", enforcement=level, active_mode="agent"),
                level,
            )

    def test_ask_mode_gets_no_guidance_at_any_level(self):
        # Ask is answer-only: nothing is planned and nothing is edited, so no
        # guidance category has anything to guard at any enforcement level.
        for level in ("strict", "light", "off"):
            for category in _ALL_GUIDANCE:
                self.assertFalse(
                    _guidance_enabled(category, enforcement=level, active_mode="ask"),
                    f"{category} @ {level}",
                )

    def test_callables_present(self):
        for n in _CORE_NUDGES:
            self.assertTrue(callable(n.should_fire))
            self.assertTrue(callable(n.render))


if __name__ == "__main__":
    unittest.main()
