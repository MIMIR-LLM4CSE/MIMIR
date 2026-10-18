from mimir.tests.support.context import build_execution_context
import unittest

from mimir.client.config.limits import EXERCISE_BUDGET
from mimir.client.guardrails.nudges.cascade import CORE_NUDGES, CoreNudge
from mimir.client.guardrails.nudges.tiers import ALL_GUIDANCE, guidance_enabled
from mimir.client.guardrails.nudges.plugins import VALID_NUDGE_LAYERS


class CoreNudgeTableTest(unittest.TestCase):
    def test_all_rows_are_core_nudges(self):
        self.assertTrue(all(isinstance(n, CoreNudge) for n in CORE_NUDGES))

    def test_names_unique(self):
        names = [n.name for n in CORE_NUDGES]
        self.assertEqual(len(names), len(set(names)))

    def test_layers_valid(self):
        for n in CORE_NUDGES:
            self.assertIn(n.layer, VALID_NUDGE_LAYERS)

    def test_verification_before_guidance(self):
        # Ordering IS priority; verification rows must all precede guidance rows.
        layers = [n.layer for n in CORE_NUDGES]
        first_guidance = layers.index("guidance")
        self.assertNotIn("verification", layers[first_guidance:])

    def test_verification_categories(self):
        verif = {n.name for n in CORE_NUDGES if n.layer == "verification"}
        self.assertEqual(
            verif, {"denial", "error_recovery", "stuck_repair", "validation",
                    "regression", "unexercised", "unfinished_plan"}
        )

    def test_no_row_asks_for_a_verdict(self):
        # A verdict is asked for in-band (the VERDICT_DUE annotation on the run's own
        # result) and reported by the ledger when it never came. As a turn-end row it
        # fired on every green run, throwing away a finished answer for a label.
        self.assertNotIn("output_verdict", {n.name for n in CORE_NUDGES})

    def test_the_exercise_rows_share_one_budget(self):
        # "run the existing test" and "nothing was exercised" are one question; two
        # budgets made it two re-prompts.
        shared = {n.name for n in CORE_NUDGES if n.budget_key == EXERCISE_BUDGET}
        self.assertEqual(shared, {"regression", "unexercised"})

    def test_validation_is_the_only_required_row_of_the_three(self):
        # It precedes them, and it is the one nothing lets go of: a file the model
        # modified and never checked blocks the conclusion, an unrun one does not.
        names = [n.name for n in CORE_NUDGES]
        for advisory in ("regression", "unexercised"):
            self.assertLess(names.index("validation"), names.index(advisory))
        self.assertEqual(
            CORE_NUDGES[names.index("validation")].budget_key, "",
        )

    def test_verification_layer_is_never_enforcement_gated(self):
        # Verification rows check reality (disk/process state), not the model's
        # reasoning, so none of them may leak into the guidance table — that table
        # is the only thing enforcement can switch off. A verification nudge that
        # appeared in ALL_GUIDANCE would silently go dead at enforcement="off".
        verif = {n.name for n in CORE_NUDGES if n.layer == "verification"}
        self.assertEqual(verif & set(ALL_GUIDANCE), set())

    def test_guidance_categories_match_guidance_table(self):
        # Every guidance category gated by _GUIDANCE_BY_LEVEL_MODE has exactly one row.
        guidance = {n.name for n in CORE_NUDGES if n.layer == "guidance"}
        self.assertEqual(guidance, set(ALL_GUIDANCE))

    def test_validation_is_not_enforcement_gated(self):
        # Checking a file one just modified is the working order's one requirement,
        # not a reasoning shim to dial down: it must survive enforcement="off".
        self.assertNotIn("validation", ALL_GUIDANCE)
        for level in ("strict", "light", "off"):
            self.assertFalse(
                guidance_enabled("validation", enforcement=level, active_mode="agent"),
                level,
            )

    def test_ask_mode_gets_no_guidance_at_any_level(self):
        # Ask is answer-only: nothing is planned and nothing is edited, so no
        # guidance category has anything to guard at any enforcement level.
        for level in ("strict", "light", "off"):
            for category in ALL_GUIDANCE:
                self.assertFalse(
                    guidance_enabled(category, enforcement=level, active_mode="ask"),
                    f"{category} @ {level}",
                )

    def test_callables_present(self):
        for n in CORE_NUDGES:
            self.assertTrue(callable(n.should_fire))
            self.assertTrue(callable(n.render))


if __name__ == "__main__":
    unittest.main()


class UnwatchedRunNudgeTest(unittest.TestCase):
    """A job submitted outside MIMIR is watched by nothing; the nudge asks for the one call
    that fixes it before the turn ends.
    """

    def _context(self, **over):
        ctx = build_execution_context(unwatched_runs={"15306620"}, workflow_state="conclude", nudge_counts={})
        for _name, _value in over.items():
            setattr(ctx.home(_name), _name, _value)
        return ctx

    def _fires(self, ctx, *, level="strict", mode="agent") -> bool:
        from mimir.client.guardrails.nudges.predicates import should_nudge_unwatched_run
        return should_nudge_unwatched_run(ctx, level=level, active_mode=mode)

    def test_it_fires_on_a_recorded_unwatched_job_at_the_conclusion(self):
        self.assertTrue(self._fires(self._context()))

    def test_it_waits_for_the_conclusion(self):
        # Mid-run the job may be about to be adopted by the very next call.
        self.assertFalse(self._fires(self._context(workflow_state="execute")))

    def test_nothing_unwatched_is_nothing_to_say(self):
        self.assertFalse(self._fires(self._context(unwatched_runs=set())))

    def test_it_survives_the_light_carve_out(self):
        # The carve-out keeps only what guards a costly, hard-to-detect,
        # non-self-correcting mistake. A lost night is all three.
        self.assertTrue(self._fires(self._context(), level="light"))

    def test_it_is_budgeted_once(self):
        from mimir.client.config.limits import NUDGE_MAX_UNWATCHED_RUN
        ctx = self._context(nudge_counts={"unwatched_run": NUDGE_MAX_UNWATCHED_RUN})
        self.assertFalse(self._fires(ctx))

    def test_ask_mode_has_no_runs_to_hand_over(self):
        self.assertFalse(self._fires(self._context(), mode="ask"))

    def test_the_message_names_the_ids_and_both_handles(self):
        from mimir.client.guardrails.nudges.messages import unwatched_run_nudge_message
        text = unwatched_run_nudge_message(
            build_execution_context(unwatched_runs={"15306620", "15306621"}))
        self.assertIn("15306620", text)
        self.assertIn("15306621", text)
        # Both kinds, because the set is kind-agnostic by construction.
        self.assertIn("slurm_watch", text)
        self.assertIn("bash_watch", text)
        # And the way out, so a job that is deliberately somebody else's is not adopted
        # to silence a reminder.
        self.assertIn("not yours to follow", text)
