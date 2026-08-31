"""The end-of-run honesty surface: what the machine records vs what the model says.

Before this existed, a run's closing prose and the recorded evidence sat side by
side with nothing reconciling them — so "verified and working" could be emitted
directly above a file that had only ever been executed, never checked against
anything, and above a checklist with most of its boxes still open.

Three separate mechanisms are pinned here:
  * the verification ledger appended to every answer (`finalize`),
  * the completion issues/completed lists that feed the incomplete-answer
    finalizer (`workflow`),
  * `needs_incomplete_finalization`, the predicate that decides whether that
    finalizer runs at all (`nudges.engine`).

Everything degrades to today's behaviour when there is no checklist, which is the
majority of runs — see the no-plan tests in each class.

Pure-Python + temp files (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from mimir.client.config.constants import EXERCISE_BUDGET
from mimir.client.context.capabilities import CODE_EXEC, JUDGE, ToolCaps
from mimir.client.context.execution_context import (
    build_execution_context,
    raise_validation_tier,
    record_run,
)
from mimir.client.event_sink import reset_event_sink, set_event_sink
from mimir.client.guardrails.nudges.engine import needs_incomplete_finalization
from mimir.client.guardrails.nudges import maybe_append_nudge
from mimir.client.guardrails.nudges.messages import unexercised_code_nudge_message
from mimir.client.guardrails.verdict import apply_verdict
from mimir.client.guardrails.workflow import (
    HEADLINE_HANDBACK,
    HEADLINE_INCOMPLETE,
    HEADLINE_REFUSED_ONLY,
    TERMINATION_STEP_LIMIT,
    TERMINATION_USER_STOPPED,
    _collect_completion_issues,
    finalize_incomplete_answer,
    is_incomplete_answer,
    unchecked_checklist_items,
    unjudged_run_lines,
)
from mimir.client.query_engine.finalize import _annotate_answer_with_changes
from mimir.client.query_engine.verification import (
    LEDGER_MARKER,
    build_ledger,
    parse_ledger_block,
    split_answer_ledger,
)


def _ctx(**over):
    ec = build_execution_context()
    ec.update(over)
    return ec


def _written(paths, tier=None, validated=True):
    ec = _ctx(
        dirty_written_files=set(paths),
        validated_files=set(paths) if validated else set(),
        code_mutation_started=True,
    )
    if validated and tier:
        for p in paths:
            raise_validation_tier(ec, p, tier)
    return ec


class _ChecklistFixture(unittest.TestCase):
    """Writes a real todo_list.md, since the readers parse the file from disk."""

    def checklist(self, ec, lines):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "todo_list.md")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self.addCleanup(lambda: (os.remove(fp), os.rmdir(d)))
        ec["todo_file_path"] = fp
        return fp


class VerificationLedgerTests(_ChecklistFixture):
    def test_no_work_no_ledger(self):
        self.assertEqual(_annotate_answer_with_changes("Done.", _ctx()), "Done.")

    def test_unchecked_file_is_named_as_such(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, validated=False))
        self.assertIn("Verification ledger", out)
        self.assertIn("`a.py` — **not checked**", out)

    def test_tier_is_reported_per_file(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="static"))
        self.assertIn("`a.py` — checked: static", out)

    def test_a_checked_file_is_not_reported_as_a_correct_one(self):
        # The wave2d case: the model's prose claims verification; the ledger must say
        # in the same breath that a checker never looked at the answer.
        out = _annotate_answer_with_changes(
            "All done, verified and working.", _written({"solver.py"}, tier="static"),
        )
        self.assertIn("says nothing about whether the answer is right", out)

    def test_the_caveat_is_domain_neutral(self):
        """The line fires on every run, so it must not speak numerics.

        Naming reference comparisons, conservation checks and convergence measurements
        made it unsatisfiable for a parser or a CLI: it printed on every such run and
        became wallpaper.
        """
        out = _annotate_answer_with_changes("Done.", _written({"parser.py"}, tier="static"))
        for word in ("conservation", "convergence", "reference comparison"):
            self.assertNotIn(word, out, f"{word!r} leaked into the neutral caveat")

    def test_the_caveat_never_points_at_rows_that_are_not_there(self):
        # It only ever prints when nothing ran, so a pointer to "the run rows below"
        # sent the reader to an empty half of the ledger — the way a caveat stops being
        # read at all.
        out = _annotate_answer_with_changes("Done.", _written({"solver.py"}, tier="static"))
        self.assertIn("nothing here was built or run", out)
        self.assertNotIn("rows below", out)

    def test_a_judged_run_replaces_the_caveat(self):
        # Once a run has been read, the ledger has something better to show than the
        # reminder that a checker proves nothing about the answer.
        ec = _written({"solver.py"}, tier="static")
        ec["runs"] = {"pytest -q": {
            "completed": True, "verdict": "pass", "reason": "l2_rel=3e-4",
            "failures": 0, "attempts": [],
        }}
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertNotIn("says nothing about whether the answer is right", out)
        self.assertIn("`pytest -q` — ran; verdict: pass — l2_rel=3e-4", out)

    def test_declared_but_never_written_is_reported(self):
        ec = _written({"a.py"}, tier="static")
        ec["declared_edit_set"] = {"a.py", "b.py"}
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("**Declared but never written:** `b.py`", out)

    def test_declared_target_written_under_another_spelling_is_not_reported(self):
        # The two sides are spelled differently by construction: a write outside the
        # workspace records an absolute path, while the checklist prose names the file
        # bare ("write wave_solver_2d.py in ../other/") or relative to the root. All
        # three spellings are the same promise, kept.
        import mimir.client.config.constants as constants
        import os
        root = constants.WORKSPACE_ROOT
        outside = os.path.abspath(os.path.join(root, "../other/wave_solver_2d.py"))
        for declared in ("wave_solver_2d.py", "../other/wave_solver_2d.py", outside):
            ec = _written({outside}, tier="static")
            ec["declared_edit_set"] = {declared}
            out = _annotate_answer_with_changes("Done.", ec)
            self.assertNotIn("Declared but never written", out, declared)

    def test_declared_elsewhere_is_still_reported(self):
        # Basename matching is only for a *bare* mention, which carried no location.
        # A declared path that names a directory must still resolve to the same file.
        import mimir.client.config.constants as constants
        import os
        root = constants.WORKSPACE_ROOT
        ec = _written({os.path.abspath(os.path.join(root, "src/solver.py"))}, tier="static")
        ec["declared_edit_set"] = {"tests/solver.py"}
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("**Declared but never written:** `tests/solver.py`", out)

    def test_unchecked_steps_are_reported_with_a_preview(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] write solver", "- [ ] add convergence test",
                            "- [ ] document it"])
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("2 steps unchecked", out)
        self.assertIn("add convergence test", out)

    def test_optional_steps_are_counted_separately(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] write solver", "- [ ] (optional) convergence study"])
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("1 optional step not done", out)
        self.assertNotIn("unchecked", out)

    def test_no_checklist_yields_no_checklist_lines(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="static"))
        self.assertNotIn("Checklist", out)

    def test_ledger_never_replaces_the_model_answer(self):
        out = _annotate_answer_with_changes("Prose.", _written({"a.py"}, tier="static"))
        self.assertTrue(out.startswith("Prose."))


class LedgerMarkerTests(_ChecklistFixture):
    """The marker contract the front-ends collapse the ledger on.

    Both UIs lift the block off the answer and render it as a disclosure panel, so the
    marker must survive the round-trip and the answer prose must come back untouched.
    """

    def test_block_is_split_off_leaving_the_prose_intact(self):
        out = _annotate_answer_with_changes("Prose.\n\nMore prose.", _written({"a.py"}))
        body, block = split_answer_ledger(out)
        self.assertEqual(body, "Prose.\n\nMore prose.")
        self.assertTrue(block.startswith(LEDGER_MARKER))

    def test_an_answer_without_a_ledger_splits_to_itself(self):
        body, block = split_answer_ledger("Just prose.")
        self.assertEqual(body, "Just prose.")
        self.assertIsNone(block)

    def test_header_fields_round_trip_through_the_marker(self):
        ec = _written({"a.py", "b.py"}, validated=False)
        _, block = split_answer_ledger(_annotate_answer_with_changes("Done.", ec))
        parsed = parse_ledger_block(block)
        self.assertEqual(parsed["status"], "warn")
        self.assertEqual(parsed["files"], 2)
        self.assertIn("2 not checked", parsed["summary"])
        self.assertEqual(len(parsed["rows"]), 2)

    def test_status_separates_clean_runs_soft_caveats_and_gaps(self):
        # ok: settled evidence, nothing open. note: it checks out, but nothing read the
        # result. warn: something the reader has to act on.
        clean = _written({"a.py"}, tier="static")
        clean["runs"] = {"pytest -q": {
            "completed": True, "verdict": "pass", "reason": "l2_rel=3e-4",
            "failures": 0, "attempts": [],
        }}
        self.assertEqual(build_ledger(clean)["status"], "note")
        self.assertEqual(build_ledger(_written({"a.py"}, tier="static"))["status"], "note")
        self.assertEqual(build_ledger(_written({"a.py"}, validated=False))["status"], "warn")

    def test_why_nothing_ran_is_reported_not_merely_omitted(self):
        # Suppressing the run recommendation and suppressing the fact that it could not
        # be followed are different things; the second is what the reader needs.
        ec = _written({"a.py"}, tier="static")
        ec["exercise_blocked_reason"] = "no execution tool is connected to this session"
        rows = "\n".join(build_ledger(ec)["rows"])
        self.assertIn("no execution tool is connected", rows)
        # Silent when nothing blocked it: an ordinary unexercised change says enough.
        self.assertNotIn(
            "out of reach", "\n".join(build_ledger(_written({"a.py"}, tier="static"))["rows"]),
        )

    def test_only_rows_needing_action_are_emphasised(self):
        """Bold is what the webview tints rows by, so it must mark exactly the gaps."""
        settled = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="static"))
        self.assertNotIn("**", settled)
        ec = _written({"a.py"}, validated=False)
        self.checklist(ec, ["- [ ] add the test"])
        for row in parse_ledger_block(split_answer_ledger(
            _annotate_answer_with_changes("Done.", ec))[1])["rows"]:
            self.assertIn("**", row, row)


class OutputVerdictLedgerTests(_ChecklistFixture):
    """Machine-observed and model-asserted, side by side and never conflated.

    A file row says what a checker established; a run row says what happened when the
    code was executed and what the model read in the output. The ledger has to render
    four run states: never judged, judged pass, judged fail, judged unknown — plus a run
    that never completed at all.
    """

    def _run(self, command="python b_test.py", **over):
        run = {"completed": True, "verdict": "", "reason": "",
               "failures": 0, "attempts": [], "call_id": ""}
        run.update(over)
        return _ctx(runs={command: run})

    def test_unjudged_output_is_named_as_such(self):
        out = _annotate_answer_with_changes("All done, verified and working.", self._run())
        self.assertIn("`python b_test.py` — ran; **its output was never judged**", out)

    def test_a_run_left_unresolved_is_reported_as_judged_unknown(self):
        # After two reminders the loop stops asking — what it must never do is let the
        # run vanish, which would read back as a clean session.
        ec = self._run("python solver.py", verdict="unknown",
                       reason="no reference for this regime")
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn(
            "`python solver.py` — ran; **judged unknown** — no reference for this regime", out,
        )

    def test_a_run_with_no_file_still_earns_a_ledger(self):
        # An analysis-only session is exactly the one whose whole answer rests on that
        # output — it used to produce no ledger at all.
        out = _annotate_answer_with_changes("The suite is green.", self._run("pytest -q"))
        self.assertIn("`pytest -q` — ran; **its output was never judged**", out)

    def test_a_run_that_never_completed_is_reported_as_such(self):
        out = _annotate_answer_with_changes("Done.", self._run(completed=False))
        self.assertIn("`python b_test.py` — **did not complete**", out)

    def test_a_failing_verdict_is_reported_with_its_reason(self):
        ec = self._run(verdict="fail", reason="0.00% amplitude reduction")
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("**verdict: fail** — 0.00% amplitude reduction", out)

    def test_a_verdict_is_labelled_as_the_models_own_reading(self):
        ec = self._run("pytest", verdict="pass", reason="matches the reference")
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("`pytest` — ran; verdict: pass — matches the reference", out)
        self.assertIn("the model's own reading", out)

    def test_what_was_tried_is_reported_when_the_budget_runs_out(self):
        from mimir.client.config.constants import VALIDATION_RETRY_BUDGET
        ec = self._run(
            verdict="fail", reason="energy grew by 0.6%",
            failures=VALIDATION_RETRY_BUDGET,
            attempts=["0.00% amplitude reduction", "energy grew by 0.6%"],
        )
        ec["dirty_written_files"] = {"solver.py"}
        ec["validated_files"] = {"solver.py"}
        ec["code_mutation_started"] = True
        issues, _ = _collect_completion_issues(ec)
        joined = "\n".join(issues)
        self.assertIn("budget exhausted", joined)
        self.assertIn("0.00% amplitude reduction", joined)
        self.assertIn("energy grew by 0.6%", joined)

    def test_the_report_never_promises_a_retry(self):
        # The loop is over by the time this is rendered; a retry budget with room
        # left describes what the loop *would* have done, not what will happen.
        ec = self._run(verdict="fail", reason="segfault", failures=1)
        ec["dirty_written_files"] = {"solver.py"}
        ec["validated_files"] = {"solver.py"}
        ec["code_mutation_started"] = True
        joined = "\n".join(_collect_completion_issues(ec)[0])
        self.assertIn("unresolved", joined)
        self.assertNotIn("will retry", joined)


class WorkflowNudgeSequenceTests(unittest.TestCase):
    """The verification nudges are one workflow, and never speak over each other.

    Write → check → run. Each step has exactly one nudge that owns it, and the
    predicates are written so that at most one can fire at a time: `validation` while a
    file still owes a check, `unexercised` once the checks are green and nothing has
    run. The gap this pins is the second one — before it existed, a model could write
    code, lint it, and stop, and the whole judging machinery stayed unreachable because
    it is only ever entered by running something. Judging is where the sequence ends:
    no nudge asks for a verdict.
    """

    def _agent(self, *, exec_tool=True, judge_tool=True):
        reg = {}
        if exec_tool:
            reg["bash_run"] = ToolCaps(name="bash_run", capabilities=frozenset({CODE_EXEC}))
        if judge_tool:
            reg["report_verdict"] = ToolCaps(
                name="report_verdict", capabilities=frozenset({JUDGE}))
        return types.SimpleNamespace(tool_caps=reg, enforcement="strict", model="test")

    def _fired(self, ec, *, agent=None):
        """The category of the one nudge this call fires, or "" for none.

        Read off the emitted event rather than the counters: the advisory rows ration
        one shared budget, so a counter diff would name the budget instead of the
        reminder that actually spoke.
        """
        # Two distinct discovery signals, else the guidance layer's `discovery` nudge
        # speaks first and this reads as "no workflow nudge fired".
        ec.setdefault("read_files", set()).add("solver.py")
        ec["searched"] = True
        events: list[dict] = []
        token = set_event_sink(events.append)
        try:
            maybe_append_nudge(
                agent=agent or self._agent(), query="fix the solver",
                active_mode="agent", execution_context=ec, messages=[],
            )
        finally:
            reset_event_sink(token)
        return next(
            (e["category"] for e in events if e.get("type") == "nudge_injected"), "",
        )

    def test_each_step_of_the_workflow_has_exactly_one_nudge(self) -> None:
        unchecked = _written({"solver.py"}, validated=False)
        # `validation` now carries a finding rather than a request, so it fires on the
        # built-in check having rejected the file, not on editing having paused.
        unchecked["builtin_check_findings"] = {"solver.py": "line 3: '{' is never closed"}
        cases = [
            ("written, never checked", unchecked, "validation"),
            ("checked, never run", _written({"solver.py"}, tier="static"), "unexercised"),
        ]
        for label, ec, expected in cases:
            with self.subTest(step=label):
                self.assertEqual(self._fired(ec), expected)

        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=True)
        self.assertEqual(
            self._fired(ec), "",
            "a run that happened ends the sequence — nothing asks for the verdict",
        )

        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=True)
        apply_verdict("pass", "l2_rel=3e-4 against the analytic solution", "", ec)
        self.assertEqual(self._fired(ec), "", "a judged run leaves nothing to ask")

    def test_a_run_of_any_kind_silences_the_execution_reminder(self) -> None:
        # Including one that failed: the model is already on the repair ladder, and
        # "you never ran it" would be false as well as unhelpful.
        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=False)
        self.assertNotEqual(self._fired(ec), "unexercised")

    def test_it_stays_silent_with_no_execution_surface_connected(self) -> None:
        # Same rule the verdict nudge follows: advice with nowhere to land is noise.
        ec = _written({"solver.py"}, tier="static")
        self.assertEqual(self._fired(ec, agent=self._agent(exec_tool=False)), "")

    def test_it_stays_silent_when_running_it_would_take_a_build_first(self) -> None:
        # A recommendation, not a requirement: exercising a CUDA kernel means a
        # toolchain and a GPU, and asking for it here only teaches the model to talk
        # its way past the reminder.
        ec = _written({"kernel.cu"}, tier="static")
        self.assertEqual(self._fired(ec), "")

    def test_it_stays_silent_when_the_environment_could_not_even_import(self) -> None:
        # The env-resolution advice owns this moment; "you never ran it" would be
        # asking for the run that just failed to start.
        ec = _written({"solver.py"}, tier="static")
        ec["unresolved_modules"] = {"numpy"}
        self.assertNotEqual(self._fired(ec), "unexercised")

    def test_an_unrun_change_still_concludes(self) -> None:
        # The whole point of the split: the check is required, the run is recommended.
        ec = _written({"solver.py"}, tier="static")
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_a_discovery_only_turn_is_never_told_to_run_something(self) -> None:
        self.assertEqual(self._fired(build_execution_context()), "")

    def test_it_asks_once_and_then_lets_go(self) -> None:
        # "There was nothing to run" is an answer the message explicitly invites, so
        # repeating the question would be arguing with a legitimate reply.
        ec = _written({"solver.py"}, tier="static")
        agent = self._agent()
        for _ in range(3):
            messages: list[dict] = []
            maybe_append_nudge(agent=agent, query="fix it", active_mode="agent",
                               execution_context=ec, messages=messages)
        self.assertEqual(ec["nudge_counts"][EXERCISE_BUDGET], 1)

    def test_the_reminder_names_the_three_routes_and_the_way_out(self) -> None:
        text = unexercised_code_nudge_message(["solver.py"])
        self.assertIn("solver.py", text)
        for route in ("entry point", "calls into it", "test directory"):
            self.assertIn(route, text)
        self.assertIn("nothing to run", text)


class CompletionIssueTests(_ChecklistFixture):
    def test_checked_sentence_does_not_claim_correctness(self):
        # The unqualified sentence is what a model reads back as licence to report
        # the work as verified.
        _, completed = _collect_completion_issues(_written({"a.py"}, tier="static"))
        self.assertIn(
            "All modified files checked (static) — parses, says nothing about the result",
            completed,
        )

    def test_weakest_tier_governs_a_multi_file_change(self):
        ec = _written({"a.py", "b.py"})
        raise_validation_tier(ec, "a.py", "static")
        raise_validation_tier(ec, "b.py", "syntax")
        _, completed = _collect_completion_issues(ec)
        # The value reported is the floor across the change, and the sentence must say
        # so: it used to print the weakest tier under the word "highest".
        self.assertIn("checked (syntax)", " ".join(completed))
        self.assertNotIn("highest evidence", " ".join(completed))

    def test_unchecked_checklist_becomes_an_issue(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [ ] add the test"])
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("Checklist incomplete" in i for i in issues))

    def test_optional_steps_are_not_an_issue(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [ ] [optional] tune the sponge"])
        issues, _ = _collect_completion_issues(ec)
        self.assertFalse(any("Checklist incomplete" in i for i in issues))

    def test_declared_but_unwritten_becomes_an_issue(self):
        ec = _written({"a.py"}, tier="static")
        ec["declared_edit_set"] = {"a.py", "b.py"}
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("Declared but never written" in i for i in issues))


class IncompleteFinalizationTests(_ChecklistFixture):
    def test_all_validated_with_no_checklist_is_complete(self):
        # Unchanged behaviour for the majority of runs.
        self.assertFalse(needs_incomplete_finalization(_written({"a.py"}, tier="static")))

    def test_all_validated_but_checklist_open_is_incomplete(self):
        # The gap this closes: validating the two files you wrote is no evidence
        # about the three steps you never started.
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] one", "- [ ] two", "- [ ] three"])
        self.assertTrue(needs_incomplete_finalization(ec))

    def test_open_checklist_without_code_mutation_is_not_forced(self):
        ec = _ctx()
        self.checklist(ec, ["- [ ] think about it"])
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_optional_steps_alone_do_not_force_it(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] one", "- [ ] optional: extra benchmark"])
        self.assertFalse(needs_incomplete_finalization(ec))


class RecommendedAxesDoNotBlockTests(_ChecklistFixture):
    """Only the check axis blocks. Build and run are recommended, and stay that way.

    `workflow_state` used to be a third condition in the gate, which quietly made the
    recommendation mandatory: a failed run — or a `fail` verdict, which drives the same
    ladder — sends the state machine back to `edit`, so every answer came back "Task is
    incomplete" until the run had failed VALIDATION_RETRY_BUDGET times. That is what
    made the model keep insisting on a validation it had been told was optional.
    """

    def test_a_failing_run_does_not_make_a_checked_change_incomplete(self):
        from mimir.client.guardrails.observations import _register_run_failure
        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=False)
        _register_run_failure(ec, "python solver.py", "segfault")
        # The state machine still steers the model back to the code — it just no
        # longer doubles as the completion gate.
        self.assertEqual(ec["workflow_state"], "edit")
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_a_fail_verdict_does_not_either(self):
        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=True)
        apply_verdict("fail", "energy grows", "", ec)
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_an_unjudged_run_is_reported_without_blocking(self):
        # Reported, never charged: it gets its own report section, stays out of the
        # issue list that decides the headline, and never reaches the gate. Asking
        # for a verdict is a recommendation, so its absence is a gap in the record
        # and not a defect in the change.
        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=True)
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("never judged" in line for line in unjudged_run_lines(ec)))
        self.assertFalse(any("never judged" in i for i in issues))
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_an_unknown_verdict_is_reported_without_blocking(self):
        # `unknown` is documented as the honest answer for output that settles
        # nothing; charging it as an issue contradicted that in the same breath.
        ec = _written({"solver.py"}, tier="static")
        record_run(ec, "python solver.py", completed=True)
        apply_verdict("unknown", "only prints 'done'", "", ec)
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("inconclusive" in line for line in unjudged_run_lines(ec)))
        self.assertFalse(any("inconclusive" in i for i in issues))
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_a_file_nothing_here_can_check_still_concludes(self):
        # The Fortran-without-gfortran case: no check was asked for, so declaring the
        # task incomplete over it is the loop blaming the model for its own silence.
        ec = _ctx(
            dirty_written_files={"model.f90"},
            unverifiable_files={"model.f90"},
            code_mutation_started=True,
            workflow_state="edit",
        )
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_an_unchecked_file_still_blocks(self):
        self.assertTrue(needs_incomplete_finalization(_written({"a.py"}, validated=False)))


class RefusedActionReportTests(_ChecklistFixture):
    """How a refused approval reads in the closing report.

    A refusal used to be filed as a blocker unconditionally, so "the user told me
    that step was unnecessary" and "the user stopped me" produced the same verdict:
    `Task is incomplete.` at high risk. The three readings of a refusal have three
    different honest endings, and none of them is silence about the skipped step.
    """

    def _refused(self, times=1, **over):
        # An otherwise clean run — validated, concluded, nothing else outstanding — so
        # the refusal is the only thing the report has to account for.
        ec = _written({"a.py"}, tier="static")
        ec["workflow_state"] = "conclude"
        ec.update(over)
        for _ in range(times):
            ec["denied_tool_calls"].append(
                {"tool": "bash_run", "path": "", "scope": "bash:bash_run:pip install"})
            ec["denial_history"].append(
                {"tool": "bash_run", "scope": "bash:bash_run:pip install", "kind": "denied"})
        return ec

    def test_a_dropped_step_is_reported_but_is_not_a_failure(self):
        out = finalize_incomplete_answer("Done.", self._refused())
        self.assertTrue(out.startswith(HEADLINE_REFUSED_ONLY))
        self.assertIn("Not performed (you refused these", out)
        self.assertIn("bash_run", out)
        # Named and visible, but not filed as something still to fix.
        self.assertNotIn("Remaining issues", out)
        self.assertIn("Residual risk: medium.", out)

    def test_the_end_of_the_ladder_reports_a_hand_back(self):
        out = finalize_incomplete_answer("Done.", self._refused(times=3))
        self.assertTrue(out.startswith(HEADLINE_HANDBACK))
        self.assertIn("Stopped at the user's request", out)
        self.assertIn("Residual risk: high.", out)

    def test_a_refusal_alongside_a_real_blocker_stays_incomplete(self):
        ec = self._refused()
        ec["declared_edit_set"] = {"a.py", "b.py"}  # promised and never written
        out = finalize_incomplete_answer("Done.", ec)
        self.assertTrue(out.startswith(HEADLINE_INCOMPLETE))
        self.assertIn("Declared but never written", out)

    def test_a_refused_run_is_never_labelled_with_an_unknown_blocker(self):
        # The fallback issue exists for a run with no identifiable problem; a refusal
        # is a perfectly identified one, and printing both read as two separate faults.
        out = finalize_incomplete_answer("Done.", self._refused())
        self.assertNotIn("Unknown blocker", out)

    def test_running_out_of_steps_never_borrows_the_complete_headline(self):
        # The step-limit path shares this finalizer. A run that never reached a final
        # answer is unfinished whatever else the ledger says.
        out = finalize_incomplete_answer(
            "Reached the maximum number of steps without a final answer.",
            self._refused(),
            TERMINATION_STEP_LIMIT,
        )
        self.assertTrue(out.startswith(HEADLINE_INCOMPLETE))
        self.assertIn("Not performed (you refused these", out)
        self.assertIn("the step budget ran out", out)

    def test_a_user_stop_is_not_reported_as_a_step_limit(self):
        out = finalize_incomplete_answer(
            "Stopped at the step checkpoint, at your request.",
            self._refused(),
            TERMINATION_USER_STOPPED,
        )
        self.assertIn("you declined to continue", out)
        self.assertNotIn("step budget ran out", out)

    def test_only_a_hand_back_counts_as_an_unfinished_answer(self):
        self.assertTrue(is_incomplete_answer(HEADLINE_INCOMPLETE + "\n..."))
        self.assertTrue(is_incomplete_answer(HEADLINE_HANDBACK + "\n..."))
        # A run that skipped what the user refused *is* finished — the CLI must not
        # offer to re-plan it and a sub-agent must not report it as failed.
        self.assertFalse(is_incomplete_answer(HEADLINE_REFUSED_ONLY + "\n..."))


class ChecklistReaderTests(_ChecklistFixture):
    def test_missing_file_fails_closed(self):
        ec = _ctx(todo_file_path="/nonexistent/todo_list.md")
        self.assertEqual(unchecked_checklist_items(ec), [])

    def test_absent_path_fails_closed(self):
        self.assertEqual(unchecked_checklist_items(_ctx()), [])

    def test_optional_prefixes_recognised(self):
        ec = _ctx()
        self.checklist(ec, [
            "- [ ] (optional) a", "- [ ] [optional] b", "- [ ] optional: c",
            "- [ ] Optional - d", "- [ ] optionally sneaky e", "- [ ] plain f",
        ])
        by_text = {it["text"]: it["optional"] for it in unchecked_checklist_items(ec)}
        for t, expected in [
            ("(optional) a", True), ("[optional] b", True), ("optional: c", True),
            ("Optional - d", True),
            # "optionally" is a different word — must not be swallowed by the prefix.
            ("optionally sneaky e", False), ("plain f", False),
        ]:
            self.assertEqual(by_text[t], expected, t)


class UnfinishedPlanNudgeTests(_ChecklistFixture):
    """The one new nudge: verification layer, cap 1, two valid exits."""

    def _should(self, ec):
        from mimir.client.guardrails.nudges.engine import _should_nudge_unfinished_plan
        return _should_nudge_unfinished_plan(ec)

    def test_fires_on_open_steps_after_code_was_written(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] one", "- [ ] two"])
        self.assertTrue(self._should(ec))

    def test_silent_without_a_checklist(self):
        self.assertFalse(self._should(_written({"a.py"}, tier="static")))

    def test_silent_before_any_code_was_written(self):
        ec = _ctx()
        self.checklist(ec, ["- [ ] two"])
        self.assertFalse(self._should(ec))

    def test_silent_when_everything_is_ticked(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] one", "- [x] two"])
        self.assertFalse(self._should(ec))

    def test_optional_steps_alone_do_not_fire_it(self):
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [x] one", "- [ ] (optional) extra"])
        self.assertFalse(self._should(ec))

    def test_capped_at_one(self):
        from mimir.client.config.constants import NUDGE_MAX_UNFINISHED_PLAN
        ec = _written({"a.py"}, tier="static")
        self.checklist(ec, ["- [ ] two"])
        self.assertTrue(self._should(ec))
        ec["nudge_counts"]["unfinished_plan"] = NUDGE_MAX_UNFINISHED_PLAN
        self.assertFalse(self._should(ec))

    def test_message_offers_both_exits(self):
        # A nudge with only one acceptable answer is a loop. "Say so and leave it
        # unchecked" has to be as valid as "go do it".
        from mimir.client.guardrails.nudges.messages import unfinished_plan_nudge_message
        msg = unfinished_plan_nudge_message([{"text": "add the convergence test"}])
        self.assertIn("add the convergence test", msg)
        self.assertIn("Either complete them now", msg)
        self.assertIn("leave it unchecked", msg)


class WorkspaceRootIsNameableTests(unittest.TestCase):
    """The workspace root must be stated absolutely, on every query.

    Regression, observed twice. A run told to create files "outside of the codes
    directory" created them in the workspace root — which *is* `codes` — and reported
    the constraint satisfied. The root used to be disclosed inside the repo-structure
    orientation block; that block is gone, so the line stands on its own and is
    injected unconditionally. Unconditional matters: the block was built only for a
    query classified as repo-touching, which left the root unstated on exactly the
    greenfield runs where a misplaced file is least visible.
    """

    def _content(self, root: str) -> str:
        from mimir.client.prompt.system_prompt import build_system_content
        with mock.patch.dict(os.environ, {"SEARCH_ROOT": root}):
            return build_system_content(
                active_mode="agent", tool_owner={}, sensitive_tools=set(),
                memory_context_file="", todo_file="", plan_todos=None,
            )

    def test_absolute_root_is_stated_explicitly(self):
        root = os.path.join(tempfile.mkdtemp(), "codes")
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        self.assertIn(f"Workspace root (absolute): {root}", self._content(root))

    def test_root_is_stated_even_with_nothing_on_disk(self):
        # No walk, no snapshot, no classification of the query: the line is a fact
        # about where the client resolves paths, so an empty or absent workspace
        # states it exactly like a populated one.
        root = os.path.join(tempfile.mkdtemp(), "gone")
        self.assertIn(f"Workspace root (absolute): {root}", self._content(root))

    def test_prompt_points_at_the_root_for_building_absolute_paths(self):
        """The root is a *prerequisite*, not a hint.

        Placement is enforced at the tool boundary (file tools reject relative
        paths), so the prompt no longer argues about how relative paths resolve —
        it only has to tell the model where to join from. The prose that tried to
        make the inference reliable is gone; see server_files._require_abs.
        """
        content = self._content(os.getcwd())
        self.assertIn("File tools take absolute paths", content)
        # The superseded band-aids must not linger alongside the structural fix.
        self.assertNotIn("workspace root is NOT a way out of it", content)
        self.assertNotIn("Every relative path you give resolves", content)

    def test_ledger_does_not_restate_the_workspace_root(self):
        # Was a compensating disclosure for ambiguous relative paths. The model now
        # types the destination explicitly, so restating it is noise on every answer.
        ec = _written({"wave_solver_2d/solver.py"}, tier="syntax")
        out = _annotate_answer_with_changes("Created outside the codes directory.", ec)
        self.assertIn("`wave_solver_2d/solver.py` — checked: syntax", out)
        self.assertNotIn("workspace root", out)


class BlockedRunIsALimitationTests(unittest.TestCase):
    """Attempting a recommended step must not turn a finished task into a failure.

    The incentive this removes is the whole complaint: build and run are *recommended*,
    but a red exit from one was charged to the change and reported as an unfinished task,
    so the cheapest way to a clean report was to force a green run at any cost.
    """

    def _blocked(self, reason="make is not installed here", command="make"):
        ec = _ctx(
            dirty_written_files={"solver.c"},
            validated_files={"solver.c"},
            code_mutation_started=True,
        )
        ec["validation_tier_by_file"] = {"solver.c": "compiled"}
        run = record_run(ec, command, completed=False, effect="build")
        run["blocked"] = reason
        run["reason"] = reason
        return ec

    def test_a_blocked_run_is_never_a_completion_issue(self):
        issues, _ = _collect_completion_issues(self._blocked())
        self.assertEqual([i for i in issues if "make" in i], [])

    def test_attempting_a_recommended_step_cannot_make_a_finished_task_incomplete(self):
        # The acceptance criterion, first half: checks green and one build that never
        # really ran here does not make the run need an incomplete report at all.
        self.assertFalse(needs_incomplete_finalization(self._blocked()))

    def test_when_something_else_is_open_the_wall_is_reported_but_not_charged(self):
        # Second half: once a report *is* rendered for another reason, the blocked run
        # appears under its own heading and never among the issues that drive the
        # headline. This is where the wall used to be counted as a defect.
        ec = self._blocked()
        ec["dirty_written_files"].add("mesh.c")  # a file that still owes a check
        summary = finalize_incomplete_answer("Done.", ec)
        self.assertIn(
            "Not attempted (a prerequisite this environment does not have)", summary,
        )
        self.assertIn("make is not installed here", summary)
        remaining = summary.split("Remaining issues:")[1]
        self.assertNotIn("make", remaining)

    def test_the_unknown_blocker_line_never_fires_over_a_limitation(self):
        issues, _ = _collect_completion_issues(self._blocked())
        self.assertNotIn(
            "Unknown blocker; explicit completion criteria were not met", issues,
        )

    def test_the_ledger_says_not_attempted_and_stays_at_note(self):
        ledger = build_ledger(self._blocked())
        rows = "\n".join(ledger["rows"])
        self.assertIn("**not attempted**", rows)
        self.assertIn("make is not installed here", rows)
        # `warn` is what the panel colours as a gap to act on — the incentive, via the UI.
        self.assertNotEqual(ledger["status"], "warn")
        self.assertIn("1 not attempted", ledger["summary"])
        self.assertNotIn("failed", ledger["summary"])

    def test_a_genuinely_failing_run_is_still_an_issue(self):
        # The negative control: nothing here swallows a real failure.
        ec = self._blocked()
        ec["runs"]["make"]["blocked"] = ""
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("make" in i for i in issues))
        self.assertTrue(finalize_incomplete_answer("Done.", ec).startswith(HEADLINE_INCOMPLETE))


class ExerciseRouteTests(unittest.TestCase):
    """"Simply feasible" is decided here, not recited in the prompt.

    A route is one direct command against the project as it stands. Anything that needs a
    step of its own first is out of proportion, and the gate stays silent — so the model is
    never pushed toward a step it would then have to talk its way out of.
    """

    def _agent(self):
        return types.SimpleNamespace(
            tool_caps={"bash_run": ToolCaps(
                name="bash_run", capabilities=frozenset({CODE_EXEC}))},
            enforcement="strict", model="test",
        )

    def _route(self, dirty, *, seen=(), absent=()):
        from mimir.client.guardrails.nudges import engine

        ec = _ctx(dirty_written_files=set(dirty), code_mutation_started=True)
        ec["read_files"] = set(seen)
        real = engine._any_command_on_path
        engine._any_command_on_path = lambda cmds: not any(c in absent for c in cmds)
        try:
            return engine._exercise_route(self._agent(), ec), ec
        finally:
            engine._any_command_on_path = real

    def test_a_compiled_edit_with_a_configured_build_is_a_route(self):
        # The branch a `.py`/`.sh` suffix test used to refuse by category, which left
        # every compiled change with no recommendation at all.
        route, _ = self._route({"solver.c"}, seen={"Makefile"})
        self.assertIn("Makefile", route)

    def test_an_unconfigured_build_tree_is_not_a_route(self):
        # The criterion itself: a configured tree is one command, configuring one is a
        # step of its own.
        route, ec = self._route({"solver.c"}, seen={"CMakeLists.txt"})
        self.assertEqual(route, "")
        self.assertEqual(
            ec["exercise_blocked_reason"], "nothing written here starts with one direct command",
        )

    def test_a_build_file_nobody_has_seen_is_not_a_route(self):
        # Evidence only: recommending `make` against a Makefile nobody laid eyes on is
        # not obviously proportionate either.
        self.assertEqual(self._route({"solver.c"})[0], "")

    def test_a_build_driver_that_is_not_installed_is_not_a_route(self):
        self.assertEqual(
            self._route({"solver.c"}, seen={"Makefile"}, absent=("make",))[0], "",
        )

    def test_a_python_edit_with_no_interpreter_here_is_not_a_route(self):
        # The runner is asked of PATH, never assumed from the suffix.
        self.assertEqual(
            self._route({"solver.py"}, absent=("python3", "python"))[0], "",
        )

    def test_a_python_edit_this_box_can_start_is_a_route(self):
        route, _ = self._route({"solver.py"})
        self.assertIn("solver.py", route)

    def test_a_registered_suite_is_the_route_for_a_compiled_edit(self):
        # What route 1 is for a language whose test file would still have to be built:
        # the suite is registered against a configured tree, so it is one command.
        route, _ = self._route({"solver.f90"}, seen={"CTestTestfile.cmake"})
        self.assertIn("ctest", route)

    def test_a_registered_suite_outranks_a_plain_build(self):
        # A build says the code is well formed; only a run produces a result to judge.
        route, _ = self._route(
            {"solver.cpp"}, seen={"CTestTestfile.cmake", "Makefile", "CMakeCache.txt"})
        self.assertIn("ctest", route)

    def test_a_configured_tree_with_no_registered_tests_falls_back_to_the_build(self):
        # CTestTestfile.cmake is generated only when tests exist, so its absence is the
        # signal — never assume a suite that was never registered.
        route, _ = self._route({"solver.cpp"}, seen={"CMakeCache.txt"})
        self.assertIn("CMakeCache.txt", route)

    def test_a_compiled_test_source_alone_is_not_a_route(self):
        # Pairing test_solver.f90 to solver.f90 would name something that still has to be
        # built — the criterion this gate exists to apply, violated.
        self.assertEqual(self._route({"solver.f90"}, seen={"test_solver.f90"})[0], "")

    def test_the_recommendation_names_the_route_it_found(self):
        message = unexercised_code_nudge_message(["solver.c"], "the build already configured here (Makefile)")
        self.assertIn("the build already configured here (Makefile)", message)
        self.assertIn("simply feasible", message)
        # The pinned way out survives, and gains the third ending.
        self.assertIn("nothing to run", message)
        self.assertIn("All three are complete answers", message)


class TestRedRunRaisesResidualRisk(unittest.TestCase):
    """A run left red must not be reported under `Residual risk: low`.

    Observed in the wild: a report listing two `Run failing, unresolved` lines and
    closing three lines later with `Residual risk: low.` The level read only the file
    axis, so the run ledger — the axis that carries whether the thing actually works —
    contributed nothing to it.
    """

    def _clean_but_for_runs(self):
        # Everything on the file axis settled, so the runs are the only open fact.
        ec = _written({"a.py"}, tier="static")
        ec["workflow_state"] = "conclude"
        return ec

    def test_a_failing_run_lifts_low_to_medium(self):
        ec = self._clean_but_for_runs()
        record_run(ec, "pytest -q", completed=False)
        out = finalize_incomplete_answer("Done.", ec)
        self.assertIn("failing, unresolved", out)
        self.assertIn("Residual risk: medium.", out)

    def test_a_model_stated_fail_lifts_low_to_medium(self):
        ec = self._clean_but_for_runs()
        record_run(ec, "./solver", completed=True)
        ec["runs"]["./solver"]["verdict"] = "fail"
        self.assertIn("Residual risk: medium.", finalize_incomplete_answer("Done.", ec))

    def test_a_blocked_run_stays_low(self):
        # A prerequisite this box does not have is a limitation, never a defect of the
        # change — POLICY is explicit that it must not be charged against completion.
        ec = self._clean_but_for_runs()
        record_run(ec, "cmake --build .", completed=False)
        ec["runs"]["cmake --build ."]["blocked"] = "cmake is not installed here"
        out = finalize_incomplete_answer("Done.", ec)
        self.assertIn("Not attempted", out)
        self.assertIn("Residual risk: low.", out)

    def test_an_unjudged_run_stays_low(self):
        # Deliberately not charged: a verdict is a recommendation, and charging its
        # absence is the trade POLICY already refused.
        ec = self._clean_but_for_runs()
        record_run(ec, "python solver.py", completed=True)
        out = finalize_incomplete_answer("Done.", ec)
        self.assertIn("no verdict on record", out)
        self.assertIn("Residual risk: low.", out)


if __name__ == "__main__":
    unittest.main()
