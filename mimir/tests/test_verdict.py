"""The verdict: the one thing in the ledger the model wrote.

Mimir never reads a program's output for a pass/fail — output is unbounded and belongs
to whoever wrote the code. It records the model's *statement* about that output, made
through a tool call rather than a line of prose, and these tests pin the asymmetry that
statement is subject to: a model may withhold credit from itself broadly, and may only
grant it precisely.

They also pin the separation the verdict rests on: a verdict is about a **run**, and
touches no file. Whether a file parses and lints is a checker's answer, established
without reading anything; whether the result is right is the model's, and the two are
never merged into one word.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mimir.client.guardrails.observations as observations
import mimir.client.tool_execution.executor as executor
from mimir.client.config.constants import EXERCISE_BUDGET, VALIDATION_RETRY_BUDGET
from mimir.client.context.capabilities import JUDGE, ToolCaps
from mimir.client.context.execution_context import build_execution_context, record_run
from mimir.client.guardrails.verdict import apply_verdict


class ApplyVerdictTests(unittest.TestCase):
    def _ctx(self, command="pytest -q foo.py"):
        ec = build_execution_context()
        ec["dirty_written_files"] = {"foo.py"}
        ec["code_mutation_started"] = True
        record_run(ec, command, completed=True, call_id="c1")
        return ec

    def _two_runs(self):
        ec = self._ctx()
        record_run(ec, "python bench.py", completed=True)
        return ec

    def test_nothing_outstanding_means_nothing_recorded(self) -> None:
        self.assertEqual(apply_verdict("pass", "fine", "", build_execution_context()), [])

    def test_an_unrecognised_word_records_nothing(self) -> None:
        ec = self._ctx()
        self.assertEqual(apply_verdict("probably", "fine", "", ec), [])
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "")

    def test_a_pass_validates_no_file(self) -> None:
        # The whole point of the split: reading an output right says nothing about
        # whether the file parses, and a checker is what answers that.
        ec = self._ctx()
        apply_verdict("pass", "l2_rel=3e-4 against the analytic solution", "", ec)
        self.assertEqual(ec["validated_files"], set())
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "pass")

    def test_a_settled_run_is_returned_with_its_command_and_row(self) -> None:
        # What the UI badge is drawn from: the verdict lands on the row the run was on.
        ec = self._ctx()
        settled = apply_verdict("pass", "residual 3e-4", "", ec)
        self.assertEqual([(s["command"], s["call_id"]) for s in settled],
                         [("pytest -q foo.py", "c1")])

    def test_an_unscoped_pass_settles_only_the_most_recent_run(self) -> None:
        # It would otherwise credit a run the statement may never have been about. The
        # model speaks right after reading an output, so the newest run is the subject.
        ec = self._two_runs()
        settled = apply_verdict("pass", "both are within tolerance", "", ec)
        self.assertEqual([s["command"] for s in settled], ["python bench.py"])
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "")

    def test_a_re_run_becomes_the_most_recent(self) -> None:
        # Re-registering in place would leave an unscoped `pass` crediting an older run.
        ec = self._two_runs()
        record_run(ec, "pytest -q foo.py", completed=True)
        apply_verdict("pass", "green and correct", "", ec)
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "pass")
        self.assertEqual(ec["runs"]["python bench.py"]["verdict"], "")

    def test_a_re_run_keeps_the_failure_history(self) -> None:
        # The budget counts attempts at the same command; a re-run is the next attempt.
        ec = self._ctx()
        apply_verdict("fail", "off by a factor of two", "", ec)
        record_run(ec, "pytest -q foo.py", completed=True)
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["failures"], 1)
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["attempts"],
                         ["off by a factor of two"])

    def test_a_scoped_pass_settles_only_the_run_it_names(self) -> None:
        ec = self._two_runs()
        self.assertTrue(apply_verdict("pass", "matches the reference", "bench", ec))
        self.assertEqual(ec["runs"]["python bench.py"]["verdict"], "pass")
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "")

    def test_an_unreadable_scope_on_a_pass_still_credits_exactly_one_run(self) -> None:
        """A scope nobody can match is read as no scope, not as a reason to drop it.

        Discarding it recorded nothing and emitted nothing, then the reminder asked for
        the statement the model had just made — which a model answers by making it
        again, unchanged. Falling back keeps the asymmetry, because it is the direction
        that matters: a `pass` still reaches exactly one run.
        """
        ec = self._two_runs()
        settled = apply_verdict("pass", "fine", "unrelated", ec)
        self.assertEqual([r["command"] for r in settled], ["python bench.py"])
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "")

    def test_an_unreadable_scope_on_a_fail_falls_back_to_every_run(self) -> None:
        ec = self._two_runs()
        self.assertEqual(len(apply_verdict("fail", "off by two", "unrelated", ec)), 2)
        self.assertEqual({r["verdict"] for r in ec["runs"].values()}, {"fail"})

    def test_fail_settles_every_run_without_a_scope(self) -> None:
        # Withholding credit broadly is never the unsafe direction.
        ec = self._two_runs()
        self.assertEqual(len(apply_verdict("fail", "off by a factor of two", "", ec)), 2)
        self.assertEqual({r["verdict"] for r in ec["runs"].values()}, {"fail"})

    def test_a_failing_verdict_drives_the_same_ladder_as_a_red_exit(self) -> None:
        # One repair mechanism, not two: the budget, the attempt log and the workflow
        # transition are the ones a non-zero exit already feeds.
        ec = self._ctx()
        ec["workflow_state"] = "conclude"
        apply_verdict("fail", "energy grows from 1.56 to 4.02", "", ec)
        run = ec["runs"]["pytest -q foo.py"]
        self.assertEqual(run["failures"], 1)
        self.assertEqual(run["attempts"], ["energy grows from 1.56 to 4.02"])
        self.assertEqual(ec["workflow_state"], "edit")

    def test_a_run_that_cannot_be_fixed_releases_the_workflow(self) -> None:
        # Stop trying after the budget, report, move on — never wedge the loop.
        ec = self._ctx()
        for _ in range(VALIDATION_RETRY_BUDGET):
            apply_verdict("fail", "still wrong", "", ec)
            record_run(ec, "pytest -q foo.py", completed=True)
        ec["validated_files"] = {"foo.py"}
        apply_verdict("fail", "still wrong", "", ec)
        self.assertEqual(ec["workflow_state"], "conclude")

    def test_unknown_addresses_every_run_but_closes_none(self) -> None:
        # "I cannot tell" is a state somebody has to be told about at the end, so the
        # run stays outstanding carrying its verdict instead of disappearing.
        ec = self._two_runs()
        self.assertEqual(len(apply_verdict("unknown", "nothing to compare against", "", ec)), 2)
        self.assertEqual({r["verdict"] for r in ec["runs"].values()}, {"unknown"})
        self.assertEqual(ec["runs"]["python bench.py"]["reason"], "nothing to compare against")

    def test_a_later_pass_closes_a_run_left_unknown(self) -> None:
        ec = self._ctx()
        apply_verdict("unknown", "no reference for this regime", "", ec)
        apply_verdict("pass", "l2_rel=3e-4 against the analytic solution", "", ec)
        self.assertEqual(ec["runs"]["pytest -q foo.py"]["verdict"], "pass")

    def test_a_run_that_never_completed_owes_no_verdict(self) -> None:
        # Its non-zero exit is the finding; asking the model to judge output that never
        # came would be asking for a guess.
        ec = build_execution_context()
        record_run(ec, "python foo.py", completed=False)
        self.assertEqual(apply_verdict("pass", "looked fine", "", ec), [])

    def test_a_verdict_never_re_arms_the_advisory_budget(self) -> None:
        # No reminder asks for a verdict any more, so there is nothing to hand back:
        # what shares that budget is the recommendation to *run* something, and a model
        # that just judged its run does not need to be told to go and run more.
        ec = self._ctx()
        ec["nudge_counts"][EXERCISE_BUDGET] = 1
        apply_verdict("pass", "fine", "", ec)
        self.assertEqual(ec["nudge_counts"][EXERCISE_BUDGET], 1)

    def test_an_unknown_verdict_closes_the_advisory_axis(self) -> None:
        # "I cannot tell what the output showed" answers the whole question — build it,
        # run it, say what came out. Asking again is asking for a different answer.
        ec = self._ctx()
        ec["nudge_counts"][EXERCISE_BUDGET] = 1
        apply_verdict("unknown", "cannot tell", "", ec)
        self.assertEqual(ec["nudge_counts"][EXERCISE_BUDGET], 1)
        self.assertTrue(ec["exercise_advice_closed"])

    def test_settling_runs_leaves_the_budget_spent(self) -> None:
        ec = self._two_runs()
        ec["nudge_counts"][EXERCISE_BUDGET] = 1
        apply_verdict("pass", "fine", "bench.py", ec)
        apply_verdict("pass", "fine", "foo.py", ec)
        self.assertEqual(ec["nudge_counts"][EXERCISE_BUDGET], 1)


class VerdictToolTests(unittest.TestCase):
    """The channel itself: a tool call, read through the roles its server declares."""

    TOOL = "some_verdict_tool"

    def _agent(self):
        reg = {self.TOOL: ToolCaps(
            name=self.TOOL,
            capabilities=frozenset({JUDGE}),
            arg_roles={
                "verdict": ("outcome",), "verdict_reason": ("why",), "verdict_scope": ("which",),
            },
        )}
        return types.SimpleNamespace(
            tool_caps=reg,
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: p or "",
        )

    def _ctx(self):
        ec = build_execution_context()
        ec["dirty_written_files"] = {"foo.py"}
        ec["code_mutation_started"] = True
        record_run(ec, "python foo.py", completed=True, call_id="c1")
        return ec

    def _call(self, ec, args, status="ok"):
        observations.record_tool_observation(
            self._agent(), self.TOOL, args, json.dumps({"status": status}), ec,
        )

    def test_the_call_settles_the_run_through_its_declared_roles(self) -> None:
        # Neither the tool name nor its argument names are known to the client.
        ec = self._ctx()
        self._call(ec, {"outcome": "pass", "why": "l2_rel=3e-4", "which": ""})
        self.assertEqual(ec["runs"]["python foo.py"]["verdict"], "pass")
        self.assertEqual(ec["runs"]["python foo.py"]["reason"], "l2_rel=3e-4")

    def test_a_shouted_verdict_is_still_a_verdict(self) -> None:
        ec = self._ctx()
        self._call(ec, {"outcome": "PASS", "why": "residual 3e-4", "which": ""})
        self.assertEqual(ec["runs"]["python foo.py"]["verdict"], "pass")

    def test_a_failed_call_settles_nothing(self) -> None:
        ec = self._ctx()
        self._call(ec, {"outcome": "pass", "why": "x", "which": ""}, status="error")
        self.assertEqual(ec["runs"]["python foo.py"]["verdict"], "")

    def test_the_settled_row_is_announced_to_the_ui(self) -> None:
        ec = self._ctx()
        events: list[dict] = []
        original = observations.emit
        observations.emit = events.append
        try:
            self._call(ec, {"outcome": "fail", "why": "energy grows", "which": ""})
        finally:
            observations.emit = original
        self.assertEqual(events, [{"type": "verdict", "id": "c1", "verdict": "fail"}])


class VerdictDueHintTests(unittest.TestCase):
    """The just-in-time ask, attached to the run's own result rather than the prompt."""

    def _agent(self, judge_name: str | None = "judge_it"):
        reg = {}
        if judge_name:
            reg[judge_name] = ToolCaps(name=judge_name, capabilities=frozenset({JUDGE}))
        return types.SimpleNamespace(tool_caps=reg)

    def _ctx(self, commands):
        ec = build_execution_context()
        for command in commands:
            record_run(ec, command, completed=True)
        return ec

    def test_a_newly_opened_run_is_asked_about_by_the_declared_tool_name(self) -> None:
        hint = executor._build_verdict_due_hint(
            self._agent(), self._ctx(["python foo.py"]), set(),
        )
        self.assertIn("VERDICT_DUE", hint)
        self.assertIn("judge_it", hint)
        self.assertIn("python foo.py", hint)

    def test_a_run_that_was_already_open_is_not_asked_about_again(self) -> None:
        ec = self._ctx(["python foo.py"])
        self.assertEqual(
            executor._build_verdict_due_hint(self._agent(), ec, {"python foo.py"}), "",
        )

    def test_no_judging_tool_connected_means_nothing_to_ask_for(self) -> None:
        hint = executor._build_verdict_due_hint(
            self._agent(judge_name=None), self._ctx(["python foo.py"]), set(),
        )
        self.assertEqual(hint, "")


class BlockedVerdictTests(unittest.TestCase):
    """`blocked` re-imputes a red exit; it never argues with the exit code itself."""

    def _failed(self, command="make", **kw):
        ec = build_execution_context()
        ec["dirty_written_files"] = {"solver.c"}
        ec["code_mutation_started"] = True
        record_run(ec, command, completed=False, **kw)
        ec["runs"][command]["failures"] = 2
        return ec

    def test_it_returns_the_repair_budget_and_leaves_the_run_red(self) -> None:
        ec = self._failed()
        settled = apply_verdict("blocked", "no configured build tree here", "", ec)
        run = ec["runs"]["make"]
        self.assertEqual([r["command"] for r in settled], ["make"])
        self.assertEqual(run["failures"], 0)
        self.assertEqual(run["blocked"], "no configured build tree here")
        # The machine saw a red exit and that stands: nothing here reads as a success.
        self.assertFalse(run["completed"])
        self.assertEqual(run["verdict"], "blocked")

    def test_it_addresses_failed_runs_not_outstanding_ones(self) -> None:
        # The two sets are disjoint by construction, so a green run awaiting a reading is
        # never swept up by a statement about a wall.
        ec = self._failed()
        record_run(ec, "pytest -q", completed=True)
        apply_verdict("blocked", "cmake is absent", "", ec)
        self.assertEqual(ec["runs"]["pytest -q"]["verdict"], "")
        self.assertEqual(ec["runs"]["make"]["verdict"], "blocked")

    def test_it_cannot_rescue_a_completed_run(self) -> None:
        ec = build_execution_context()
        record_run(ec, "pytest -q", completed=True)
        self.assertEqual(apply_verdict("blocked", "excuse", "", ec), [])
        self.assertEqual(ec["runs"]["pytest -q"]["verdict"], "")

    def test_a_scope_naming_nothing_still_speaks_for_every_failed_run(self) -> None:
        ec = self._failed()
        record_run(ec, "cmake --build .", completed=False)
        apply_verdict("blocked", "no toolchain", "nonesuch", ec)
        self.assertEqual(
            {c for c, r in ec["runs"].items() if r["blocked"]},
            {"make", "cmake --build ."},
        )

    def test_it_closes_the_advisory_axis_like_unknown(self) -> None:
        ec = self._failed()
        apply_verdict("blocked", "no configured build tree here", "", ec)
        self.assertTrue(ec["exercise_advice_closed"])

    def test_a_blocked_run_leaves_the_repair_release_reachable(self) -> None:
        # failed_runs excludes it: left in, its permanent `failures == 0` would hold the
        # "every failure is spent" release below its threshold for the rest of the query.
        from mimir.client.context.execution_context import failed_runs

        ec = self._failed()
        apply_verdict("blocked", "no toolchain", "", ec)
        self.assertEqual(failed_runs(ec), {})


class ImputationDueHintTests(unittest.TestCase):
    """The mirror ask, on the failing result: a red exit says nothing about whose fault."""

    def _agent(self, judge_name: str | None = "judge_it"):
        reg = {}
        if judge_name:
            reg[judge_name] = ToolCaps(name=judge_name, capabilities=frozenset({JUDGE}))
        return types.SimpleNamespace(tool_caps=reg)

    def _ctx(self, command="make", completed=False):
        ec = build_execution_context()
        record_run(ec, command, completed=completed)
        return ec

    def test_a_newly_failed_run_is_asked_who_it_belongs_to(self) -> None:
        hint = executor._build_imputation_hint(self._agent(), self._ctx(), set())
        self.assertIn("IMPUTATION_DUE", hint)
        self.assertIn("judge_it", hint)
        self.assertIn("blocked", hint)
        self.assertIn("make", hint)

    def test_it_is_silent_with_no_judging_tool_connected(self) -> None:
        # Nothing to ask for: the alternative it offers would name no tool.
        self.assertEqual(
            executor._build_imputation_hint(self._agent(None), self._ctx(), set()), "",
        )

    def test_a_green_run_is_not_asked_about_here(self) -> None:
        # That half is VERDICT_DUE's; the two never fire on the same result.
        self.assertEqual(
            executor._build_imputation_hint(
                self._agent(), self._ctx(completed=True), set()), "",
        )

    def test_a_failure_already_seen_is_not_asked_about_twice(self) -> None:
        ec = self._ctx()
        self.assertEqual(
            executor._build_imputation_hint(self._agent(), ec, {"make"}), "",
        )


if __name__ == "__main__":
    unittest.main()
