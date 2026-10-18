"""The stuck-repair ladder: leaving a hypothesis the model will not leave on its own.

The signal is ``runs[cmd]["failures"]``; each rung says something different, not louder.
"""
from __future__ import annotations

import unittest


from mimir.client.config.limits import (
    STUCK_REPAIR_ADVISE_AFTER,
    STUCK_REPAIR_CONSTRAIN_AFTER,
    VALIDATION_RETRY_BUDGET,
)
from mimir.client.tools.capabilities import CODE_EXEC, JUDGE, ToolCaps
from mimir.client.context.execution_context import record_run
from mimir.tests.support.context import build_execution_context
from mimir.client.protocol.event_sink import reset_event_sink, set_event_sink
from mimir.client.guardrails.nudges import maybe_append_nudge, nudge_pending
from mimir.client.guardrails.nudges.cascade import CORE_NUDGES
from mimir.client.guardrails.nudges.predicates import (
    should_nudge_stuck_repair,
    worst_run_failure_streak,
)
from mimir.client.guardrails.nudges.messages import stuck_repair_nudge_message
from mimir.client.guardrails.verification.ladder import register_run_failure
from mimir.client.context.execution_context import run_ledger_key
from mimir.client.core.turn_state import TurnState
from mimir.tests.support.agents import fake_agent


def _ctx(command="pytest -q solver.py", failures=0, fired=0):
    """A context whose one command has already failed `failures` times."""
    ec = build_execution_context()
    ec.edits.dirty_written_files = {"solver.py"}
    ec.edits.code_mutation_started = True
    for _ in range(max(failures, 1)):
        record_run(ec, command, completed=False)
        if failures:
            register_run_failure(ec, command, "still wrong")
    if not failures:
        # Runs are keyed by identity, not by the text typed (run_ledger_key).
        key = run_ledger_key(command)
        ec.execution.runs[key]["failures"] = 0
        ec.execution.runs[key]["completed"] = True
        ec.execution.runs[key]["verdict"] = "pass"
    ec.nudges.nudge_counts["stuck_repair"] = fired
    return ec


class StreakTests(unittest.TestCase):
    def test_no_failing_run_is_no_streak(self) -> None:
        self.assertEqual(worst_run_failure_streak(build_execution_context()), 0)

    def test_the_streak_is_per_command_not_a_sum(self) -> None:
        # Two unrelated commands failing once each is ordinary early red, not a model
        # going round in circles. Summing them would fire on the honest case.
        ec = build_execution_context()
        ec.edits.code_mutation_started = True
        for cmd in ("make", "pytest -q"):
            record_run(ec, cmd, completed=False)
            register_run_failure(ec, cmd, "nope")
        self.assertEqual(worst_run_failure_streak(ec), 1)

    def test_a_re_run_keeps_climbing(self) -> None:
        # record_run carries `failures` over, which is what makes the streak survive
        # the model re-running the same command after each edit.
        ec = _ctx(failures=3)
        self.assertEqual(worst_run_failure_streak(ec), 3)

    def test_a_settled_run_leaves_the_streak(self) -> None:
        # A pass drops the run out of failed_runs, so the ladder disarms on its own.
        ec = _ctx(failures=4)
        ec.execution.runs["pytest solver.py"]["completed"] = True
        ec.execution.runs["pytest solver.py"]["verdict"] = "pass"
        self.assertEqual(worst_run_failure_streak(ec), 0)
        self.assertFalse(should_nudge_stuck_repair(ec))

    def test_a_blocked_run_is_not_a_streak(self) -> None:
        # The environment putting up a wall is not the model repeating itself, and
        # failed_runs already excludes it.
        ec = _ctx(failures=4)
        ec.execution.runs["pytest solver.py"]["blocked"] = "pytest not installed"
        self.assertEqual(worst_run_failure_streak(ec), 0)


class LadderTests(unittest.TestCase):
    def test_it_stays_quiet_below_the_first_rung(self) -> None:
        for n in range(STUCK_REPAIR_ADVISE_AFTER):
            with self.subTest(failures=n):
                self.assertFalse(should_nudge_stuck_repair(_ctx(failures=n)))

    def test_the_first_rung_speaks_on_the_second_failure(self) -> None:
        self.assertTrue(should_nudge_stuck_repair(_ctx(failures=STUCK_REPAIR_ADVISE_AFTER)))

    def test_the_first_rung_never_speaks_twice(self) -> None:
        # The regression this row exists to avoid: a streak walking 2 -> 3 must not
        # re-say "look wider". A plain `count < MAX` cap would.
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER + 1, fired=1)
        self.assertFalse(should_nudge_stuck_repair(ec))

    def test_the_second_rung_waits_for_its_own_threshold(self) -> None:
        ec = _ctx(failures=STUCK_REPAIR_CONSTRAIN_AFTER, fired=1)
        self.assertTrue(should_nudge_stuck_repair(ec))

    def test_the_second_rung_is_never_reached_before_the_first_is_spoken(self) -> None:
        # Even on a streak that jumps straight past both thresholds, rung one goes first.
        ec = _ctx(failures=STUCK_REPAIR_CONSTRAIN_AFTER + 2, fired=0)
        self.assertTrue(should_nudge_stuck_repair(ec))
        self.assertIn("failed", stuck_repair_nudge_message(worst_run_failure_streak(ec)))

    def test_there_is_no_third_rung(self) -> None:
        # Past the two rungs the existing release takes over: at
        # VALIDATION_RETRY_BUDGET failures register_run_failure moves the workflow to
        # `conclude`. Saying it a third time would just be louder.
        ec = _ctx(failures=VALIDATION_RETRY_BUDGET, fired=2)
        self.assertFalse(should_nudge_stuck_repair(ec))

    def test_the_rungs_are_ordered_inside_the_repair_window(self) -> None:
        self.assertLess(STUCK_REPAIR_ADVISE_AFTER, STUCK_REPAIR_CONSTRAIN_AFTER)
        self.assertLess(STUCK_REPAIR_CONSTRAIN_AFTER, VALIDATION_RETRY_BUDGET)


class MessageTests(unittest.TestCase):
    def test_the_two_rungs_say_different_things(self) -> None:
        advise = stuck_repair_nudge_message(STUCK_REPAIR_ADVISE_AFTER)
        constrain = stuck_repair_nudge_message(STUCK_REPAIR_CONSTRAIN_AFTER)
        self.assertNotEqual(advise, constrain)

    def test_the_first_rung_only_advises(self) -> None:
        advise = stuck_repair_nudge_message(STUCK_REPAIR_ADVISE_AFTER)
        self.assertIn("not necessarily where you are correcting it", advise)
        self.assertNotIn("Do not edit", advise)

    def test_the_second_rung_constrains_the_next_action(self) -> None:
        constrain = stuck_repair_nudge_message(STUCK_REPAIR_CONSTRAIN_AFTER)
        self.assertIn("Do not edit that target again", constrain)
        self.assertIn("check it alone", constrain)

    def test_neither_rung_proposes_a_hypothesis(self) -> None:
        # The copy must stay domain-free: what is being corrected is the refusal to
        # leave a hypothesis, not the hypothesis itself, which only the model can see.
        for streak in (STUCK_REPAIR_ADVISE_AFTER, STUCK_REPAIR_CONSTRAIN_AFTER):
            text = stuck_repair_nudge_message(streak).lower()
            for word in ("boundary condition", "off-by-one", "the bug is", "probably"):
                self.assertNotIn(word, text)

    def test_the_second_rung_names_an_acceptable_ending(self) -> None:
        # House convention: a required-feeling nudge that offers no way to close the
        # subject gets answered with another loop.
        self.assertIn("ending too", stuck_repair_nudge_message(STUCK_REPAIR_CONSTRAIN_AFTER))

    def test_the_streak_is_stated(self) -> None:
        self.assertIn("7", stuck_repair_nudge_message(7))


class TablePlacementTests(unittest.TestCase):
    def test_it_speaks_before_validation(self) -> None:
        # A model going round the same failure is not stuck on whether it finished
        # checking; only one row speaks per turn, so order is the whole decision.
        names = [n.name for n in CORE_NUDGES]
        self.assertLess(names.index("stuck_repair"), names.index("validation"))

    def test_it_rations_its_own_budget(self) -> None:
        # Not the shared exercise budget: that one is a single ask about running the
        # code, and borrowing it would silence "nothing was exercised" for good.
        row = next(n for n in CORE_NUDGES if n.name == "stuck_repair")
        self.assertEqual(row.budget_key, "")
        self.assertEqual(row.layer, "verification")


class FiringPathTests(unittest.TestCase):
    """Exercised through ``maybe_append_nudge``, which an empty render, an earlier row or the
    enforcement level could each silence.
    """

    def _agent(self):
        return fake_agent(
            tool_caps={
                "bash_run": ToolCaps(name="bash_run", capabilities=frozenset({CODE_EXEC})),
                "report_verdict": ToolCaps(
                    name="report_verdict", capabilities=frozenset({JUDGE})),
            },
            enforcement="strict", model="test",turn=TurnState())

    def _fire(self, ec):
        """The category of the one nudge this call injects, plus the text."""
        ec.discovery.read_files.add("solver.py")
        ec.discovery.searched = True
        events: list[dict] = []
        messages: list[dict] = []
        token = set_event_sink(events.append)
        try:
            maybe_append_nudge(agent=self._agent(), query="fix the solver",
                               active_mode="agent", execution_context=ec, messages=messages)
        finally:
            reset_event_sink(token)
        fired = next((e for e in events if e.get("type") == "nudge_injected"), None)
        return (fired or {}).get("category", ""), (fired or {}).get("text", "")

    def test_the_first_rung_reaches_the_model(self) -> None:
        cat, text = self._fire(_ctx(failures=STUCK_REPAIR_ADVISE_AFTER))
        self.assertEqual(cat, "stuck_repair")
        self.assertIn("not necessarily where you are correcting it", text)

    def test_the_second_rung_reaches_the_model(self) -> None:
        cat, text = self._fire(_ctx(failures=STUCK_REPAIR_CONSTRAIN_AFTER, fired=1))
        self.assertEqual(cat, "stuck_repair")
        self.assertIn("Do not edit that target again", text)

    def test_one_failure_leaves_the_other_rows_to_speak(self) -> None:
        # The ladder must not annex the ordinary first red of a repair.
        cat, _ = self._fire(_ctx(failures=1))
        self.assertNotEqual(cat, "stuck_repair")

    def test_firing_charges_its_own_counter(self) -> None:
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        self._fire(ec)
        self.assertEqual(ec.nudges.nudge_counts.get("stuck_repair"), 1)
        self.assertEqual(ec.nudges.nudge_counts.get("exercise", 0), 0)

    def test_the_probe_agrees_with_the_injection(self) -> None:
        # `nudge_pending` runs before the model call so the loop can hold streaming
        # prose; if the two walks disagreed the answer would be dropped for a reminder
        # that never came.
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        ec.discovery.read_files = {"solver.py"}
        ec.discovery.searched = True
        self.assertTrue(nudge_pending(agent=self._agent(), query="fix the solver",
                                      active_mode="agent", execution_context=ec))

    def test_the_probe_moves_no_counter(self) -> None:
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        nudge_pending(agent=self._agent(), query="fix the solver",
                      active_mode="agent", execution_context=ec)
        self.assertEqual(ec.nudges.nudge_counts.get("stuck_repair", 0), 0)

    def test_it_survives_enforcement_off(self) -> None:
        # A verification row by construction: guidance is gated off, this is not.
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        agent = self._agent()
        agent.enforcement = "off"
        ec.discovery.read_files = {"solver.py"}
        ec.discovery.searched = True
        self.assertTrue(nudge_pending(agent=agent, query="fix the solver",
                                      active_mode="agent", execution_context=ec))



class TheLadderIsReachableThroughAStatusLineTests(unittest.TestCase):
    """The ledger key holds a retried run together through a trailing ``echo "$?"``, so the
    failure counter adds up.
    """

    def test_the_failure_streak_accumulates_across_the_status_line(self) -> None:
        from mimir.client.context.execution_context import (
            record_run,
        )
        from mimir.tests.support.context import build_execution_context
        from mimir.client.guardrails.verification.ladder import register_run_failure
        base = "python3 tests/test_wave2d.py"
        forms = [base, base + '; echo "suite_exit=$?"', "cd sub && " + base]
        ec = build_execution_context()
        for cmd in forms:
            record_run(ec, cmd, completed=False)
            register_run_failure(ec, cmd, "red")
        self.assertEqual(len(ec.execution.runs), 1)
        self.assertEqual(ec.execution.runs[base]["failures"], len(forms))


if __name__ == "__main__":
    unittest.main()
