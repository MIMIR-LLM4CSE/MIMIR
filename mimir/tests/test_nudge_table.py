import unittest

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
            verif, {"denial", "error_recovery", "regression", "unexercised",
                    "unfinished_plan", "output_verdict"}
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

    def test_validation_nudge_enforcement_tier(self):
        # The validation reminder is agent-only (plan mode is read-only → nothing to
        # validate): fires at strict and light in agent mode, never in plan, never at off.
        self.assertTrue(_guidance_enabled("validation", enforcement="strict", active_mode="agent"))
        self.assertTrue(_guidance_enabled("validation", enforcement="light", active_mode="agent"))
        self.assertFalse(_guidance_enabled("validation", enforcement="strict", active_mode="plan"))
        self.assertFalse(_guidance_enabled("validation", enforcement="light", active_mode="plan"))
        self.assertFalse(_guidance_enabled("validation", enforcement="off", active_mode="agent"))
        self.assertFalse(_guidance_enabled("validation", enforcement="off", active_mode="plan"))

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
