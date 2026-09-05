"""The stuck-repair ladder: leaving a hypothesis the model will not leave on its own.

The failure this row exists for is not a wrong fix — it is the *same* wrong fix, tried
again with different letters. A model going round that loop passes every guardrail the
codebase already has: the calls are not identical (so `dispatch`'s repeat block never
arms), the edits are not identical (so `REPEATED_EDIT_FAILURE_LIMIT` never arms), and a
failing run is deliberately not a completion gate (`RecommendedAxesDoNotBlockTests`).
What is repeated is the *hypothesis*, and nothing was watching that.

These tests pin the substitute: the hypothesis is never inspected — only counted against.
`runs[cmd]["failures"]`, which `record_run` already carries across a re-run, is the whole
signal, and the row's job is to say something *different* at each rung rather than louder.
"""
from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mimir.client.config.constants import (
    NUDGE_MAX_STUCK_REPAIR,
    STUCK_REPAIR_ADVISE_AFTER,
    STUCK_REPAIR_CONSTRAIN_AFTER,
    VALIDATION_RETRY_BUDGET,
)
from mimir.client.context.capabilities import CODE_EXEC, JUDGE, ToolCaps
from mimir.client.context.execution_context import build_execution_context, record_run
from mimir.client.event_sink import reset_event_sink, set_event_sink
from mimir.client.guardrails.nudges.engine import (
    _CORE_NUDGES,
    maybe_append_nudge,
    nudge_pending,
    _should_nudge_stuck_repair,
    _worst_run_failure_streak,
)
from mimir.client.guardrails.nudges.messages import stuck_repair_nudge_message
from mimir.client.guardrails.observations import _register_run_failure


def _ctx(command="pytest -q solver.py", failures=0, fired=0):
    """A context whose one command has already failed `failures` times."""
    ec = build_execution_context()
    ec["dirty_written_files"] = {"solver.py"}
    ec["code_mutation_started"] = True
    for _ in range(max(failures, 1)):
        record_run(ec, command, completed=False)
        if failures:
            _register_run_failure(ec, command, "still wrong")
    if not failures:
        ec["runs"][command]["failures"] = 0
        ec["runs"][command]["completed"] = True
        ec["runs"][command]["verdict"] = "pass"
    ec["nudge_counts"]["stuck_repair"] = fired
    return ec


class StreakTests(unittest.TestCase):
    def test_no_failing_run_is_no_streak(self) -> None:
        self.assertEqual(_worst_run_failure_streak(build_execution_context()), 0)

    def test_the_streak_is_per_command_not_a_sum(self) -> None:
        # Two unrelated commands failing once each is ordinary early red, not a model
        # going round in circles. Summing them would fire on the honest case.
        ec = build_execution_context()
        ec["code_mutation_started"] = True
        for cmd in ("make", "pytest -q"):
            record_run(ec, cmd, completed=False)
            _register_run_failure(ec, cmd, "nope")
        self.assertEqual(_worst_run_failure_streak(ec), 1)

    def test_a_re_run_keeps_climbing(self) -> None:
        # record_run carries `failures` over, which is what makes the streak survive
        # the model re-running the same command after each edit.
        ec = _ctx(failures=3)
        self.assertEqual(_worst_run_failure_streak(ec), 3)

    def test_a_settled_run_leaves_the_streak(self) -> None:
        # A pass drops the run out of failed_runs, so the ladder disarms on its own.
        ec = _ctx(failures=4)
        ec["runs"]["pytest -q solver.py"]["completed"] = True
        ec["runs"]["pytest -q solver.py"]["verdict"] = "pass"
        self.assertEqual(_worst_run_failure_streak(ec), 0)
        self.assertFalse(_should_nudge_stuck_repair(ec))

    def test_a_blocked_run_is_not_a_streak(self) -> None:
        # The environment putting up a wall is not the model repeating itself, and
        # failed_runs already excludes it.
        ec = _ctx(failures=4)
        ec["runs"]["pytest -q solver.py"]["blocked"] = "pytest not installed"
        self.assertEqual(_worst_run_failure_streak(ec), 0)


class LadderTests(unittest.TestCase):
    def test_it_stays_quiet_below_the_first_rung(self) -> None:
        for n in range(STUCK_REPAIR_ADVISE_AFTER):
            with self.subTest(failures=n):
                self.assertFalse(_should_nudge_stuck_repair(_ctx(failures=n)))

    def test_the_first_rung_speaks_on_the_second_failure(self) -> None:
        self.assertTrue(_should_nudge_stuck_repair(_ctx(failures=STUCK_REPAIR_ADVISE_AFTER)))

    def test_the_first_rung_never_speaks_twice(self) -> None:
        # The regression this row exists to avoid: a streak walking 2 -> 3 must not
        # re-say "look wider". A plain `count < MAX` cap would.
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER + 1, fired=1)
        self.assertFalse(_should_nudge_stuck_repair(ec))

    def test_the_second_rung_waits_for_its_own_threshold(self) -> None:
        ec = _ctx(failures=STUCK_REPAIR_CONSTRAIN_AFTER, fired=1)
        self.assertTrue(_should_nudge_stuck_repair(ec))

    def test_the_second_rung_is_never_reached_before_the_first_is_spoken(self) -> None:
        # Even on a streak that jumps straight past both thresholds, rung one goes first.
        ec = _ctx(failures=STUCK_REPAIR_CONSTRAIN_AFTER + 2, fired=0)
        self.assertTrue(_should_nudge_stuck_repair(ec))
        self.assertIn("failed", stuck_repair_nudge_message(_worst_run_failure_streak(ec)))

    def test_there_is_no_third_rung(self) -> None:
        # Past the two rungs the existing release takes over: at
        # VALIDATION_RETRY_BUDGET failures _register_run_failure moves the workflow to
        # `conclude`. Saying it a third time would just be louder.
        ec = _ctx(failures=VALIDATION_RETRY_BUDGET, fired=NUDGE_MAX_STUCK_REPAIR)
        self.assertFalse(_should_nudge_stuck_repair(ec))

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
        names = [n.name for n in _CORE_NUDGES]
        self.assertLess(names.index("stuck_repair"), names.index("validation"))

    def test_it_rations_its_own_budget(self) -> None:
        # Not the shared exercise budget: that one is a single ask about running the
        # code, and borrowing it would silence "nothing was exercised" for good.
        row = next(n for n in _CORE_NUDGES if n.name == "stuck_repair")
        self.assertEqual(row.budget_key, "")
        self.assertEqual(row.layer, "verification")


class FiringPathTests(unittest.TestCase):
    """The predicate is not the feature — what reaches the model is.

    Exercised through `maybe_append_nudge` rather than the predicate alone, because the
    row has three ways to be silently inert that a predicate test cannot see: an empty
    render is skipped, an earlier verification row wins the single slot, and the whole
    guidance half is gated off at `enforcement="off"` while this half is not.
    """

    def _agent(self):
        return types.SimpleNamespace(
            tool_caps={
                "bash_run": ToolCaps(name="bash_run", capabilities=frozenset({CODE_EXEC})),
                "report_verdict": ToolCaps(
                    name="report_verdict", capabilities=frozenset({JUDGE})),
            },
            enforcement="strict", model="test",
        )

    def _fire(self, ec):
        """The category of the one nudge this call injects, plus the text."""
        ec.setdefault("read_files", set()).add("solver.py")
        ec["searched"] = True
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
        self.assertEqual(ec["nudge_counts"].get("stuck_repair"), 1)
        self.assertEqual(ec["nudge_counts"].get("exercise", 0), 0)

    def test_the_probe_agrees_with_the_injection(self) -> None:
        # `nudge_pending` runs before the model call so the loop can hold streaming
        # prose; if the two walks disagreed the answer would be dropped for a reminder
        # that never came.
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        ec["read_files"] = {"solver.py"}
        ec["searched"] = True
        self.assertTrue(nudge_pending(agent=self._agent(), query="fix the solver",
                                      active_mode="agent", execution_context=ec))

    def test_the_probe_moves_no_counter(self) -> None:
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        nudge_pending(agent=self._agent(), query="fix the solver",
                      active_mode="agent", execution_context=ec)
        self.assertEqual(ec["nudge_counts"].get("stuck_repair", 0), 0)

    def test_it_survives_enforcement_off(self) -> None:
        # A verification row by construction: guidance is gated off, this is not.
        ec = _ctx(failures=STUCK_REPAIR_ADVISE_AFTER)
        agent = self._agent()
        agent.enforcement = "off"
        ec["read_files"] = {"solver.py"}
        ec["searched"] = True
        self.assertTrue(nudge_pending(agent=agent, query="fix the solver",
                                      active_mode="agent", execution_context=ec))


if __name__ == "__main__":
    unittest.main()
