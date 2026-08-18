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
import tempfile
import unittest

from mimir.client.context.execution_context import (
    build_execution_context,
    raise_validation_tier,
)
from mimir.client.guardrails.nudges.engine import needs_incomplete_finalization
from mimir.client.guardrails.workflow import (
    HEADLINE_HANDBACK,
    HEADLINE_INCOMPLETE,
    HEADLINE_REFUSED_ONLY,
    _collect_completion_issues,
    finalize_incomplete_answer,
    is_incomplete_answer,
    unchecked_checklist_items,
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

    def test_unvalidated_file_is_named_as_such(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, validated=False))
        self.assertIn("Verification ledger", out)
        self.assertIn("`a.py` — **not validated**", out)

    def test_tier_is_reported_per_file(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="static"))
        self.assertIn("`a.py` — validated: static", out)

    def test_executed_without_an_oracle_says_so(self):
        # The wave2d case: green pytest, nothing compared. The model's prose claims
        # verification; the ledger must contradict it in the same breath.
        out = _annotate_answer_with_changes(
            "All done, verified and working.", _written({"solver.py"}, tier="executed"),
        )
        self.assertIn("Discrimination: none observed", out)
        self.assertIn("tells working code from broken code", out)

    def test_the_absence_line_is_domain_neutral(self):
        """The line fires on every non-numerical run, so it must not speak numerics.

        Naming reference comparisons, conservation checks and convergence measurements
        made it unsatisfiable for a parser or a CLI: it printed on every such run and
        became wallpaper. Numerical vocabulary is reserved for qualifying an invariant
        that was actually reported.
        """
        out = _annotate_answer_with_changes("Done.", _written({"parser.py"}, tier="executed"))
        for word in ("conservation", "convergence", "reference comparison"):
            self.assertNotIn(word, out, f"{word!r} leaked into the neutral absence line")

    def test_oracle_tier_suppresses_the_warning(self):
        out = _annotate_answer_with_changes("Done.", _written({"solver.py"}, tier="oracle"))
        self.assertNotIn("Discrimination: none observed", out)

    def test_no_oracle_line_when_nothing_was_executed(self):
        # A file that only got a syntax/static check never claimed correctness, so
        # the per-file tier already tells the whole story — adding the warning
        # would be noise on every lint-only edit.
        for tier in ("syntax", "static"):
            out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier=tier))
            self.assertNotIn("Discrimination: none observed", out, tier)

    def test_oracle_names_which_basis_earned_it(self):
        """`oracle` from a discriminating check and from a printed number differ."""
        discriminated = _written({"parser.py"}, tier="oracle")
        discriminated["executed_failures"] = {"parser.py"}
        out = _annotate_answer_with_changes("Done.", discriminated)
        self.assertIn("`parser.py` — validated: oracle (red→green)", out)
        self.assertNotIn("Reported invariant", out)

        invariant = _written({"solver.py"}, tier="oracle")
        out = _annotate_answer_with_changes("Done.", invariant)
        self.assertIn("`solver.py` — validated: oracle (reported invariant)", out)
        # An invariant is presence-only evidence; the ledger must not let it pass for
        # a comparison against something sealed.
        self.assertIn("never its value", out)

    def test_declared_but_never_written_is_reported(self):
        ec = _written({"a.py"}, tier="executed")
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
            ec = _written({outside}, tier="executed")
            ec["declared_edit_set"] = {declared}
            out = _annotate_answer_with_changes("Done.", ec)
            self.assertNotIn("Declared but never written", out, declared)

    def test_declared_elsewhere_is_still_reported(self):
        # Basename matching is only for a *bare* mention, which carried no location.
        # A declared path that names a directory must still resolve to the same file.
        import mimir.client.config.constants as constants
        import os
        root = constants.WORKSPACE_ROOT
        ec = _written({os.path.abspath(os.path.join(root, "src/solver.py"))}, tier="executed")
        ec["declared_edit_set"] = {"tests/solver.py"}
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("**Declared but never written:** `tests/solver.py`", out)

    def test_unchecked_steps_are_reported_with_a_preview(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] write solver", "- [ ] add convergence test",
                            "- [ ] document it"])
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("2 steps unchecked", out)
        self.assertIn("add convergence test", out)

    def test_optional_steps_are_counted_separately(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] write solver", "- [ ] (optional) convergence study"])
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("1 optional step not done", out)
        self.assertNotIn("unchecked", out)

    def test_no_checklist_yields_no_checklist_lines(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="oracle"))
        self.assertNotIn("Checklist", out)

    def test_ledger_never_replaces_the_model_answer(self):
        out = _annotate_answer_with_changes("Prose.", _written({"a.py"}, tier="oracle"))
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
        self.assertIn("2 not validated", parsed["summary"])
        self.assertEqual(len(parsed["rows"]), 2)

    def test_status_separates_clean_runs_soft_caveats_and_gaps(self):
        # ok: settled evidence, nothing open. note: passed, but discriminates nothing.
        # warn: something the reader has to act on.
        clean = _written({"a.py"}, tier="oracle")
        clean["executed_failures"] = {"a.py"}
        self.assertEqual(build_ledger(clean)["status"], "ok")
        self.assertEqual(build_ledger(_written({"a.py"}, tier="executed"))["status"], "note")
        self.assertEqual(build_ledger(_written({"a.py"}, validated=False))["status"], "warn")

    def test_only_rows_needing_action_are_emphasised(self):
        """Bold is what the webview tints rows by, so it must mark exactly the gaps."""
        settled = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="oracle"))
        self.assertNotIn("**", settled)
        ec = _written({"a.py"}, validated=False)
        self.checklist(ec, ["- [ ] add the test"])
        for row in parse_ledger_block(split_answer_ledger(
            _annotate_answer_with_changes("Done.", ec))[1])["rows"]:
            self.assertIn("**", row, row)


class CompletionIssueTests(_ChecklistFixture):
    def test_validated_sentence_names_the_evidence(self):
        # The unqualified sentence is what a model reads back as licence to report
        # the work as verified.
        _, completed = _collect_completion_issues(_written({"a.py"}, tier="executed"))
        self.assertIn("All modified files validated (weakest evidence: executed)", completed)

    def test_weakest_tier_governs_a_multi_file_change(self):
        ec = _written({"a.py", "b.py"})
        raise_validation_tier(ec, "a.py", "oracle")
        raise_validation_tier(ec, "b.py", "syntax")
        _, completed = _collect_completion_issues(ec)
        # The value reported is the floor across the change, and the sentence must say
        # so: it used to print the weakest tier under the word "highest".
        self.assertIn("weakest evidence: syntax", " ".join(completed))
        self.assertNotIn("highest evidence", " ".join(completed))

    def test_unchecked_checklist_becomes_an_issue(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [ ] add the test"])
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("Checklist incomplete" in i for i in issues))

    def test_optional_steps_are_not_an_issue(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [ ] [optional] tune the sponge"])
        issues, _ = _collect_completion_issues(ec)
        self.assertFalse(any("Checklist incomplete" in i for i in issues))

    def test_declared_but_unwritten_becomes_an_issue(self):
        ec = _written({"a.py"}, tier="executed")
        ec["declared_edit_set"] = {"a.py", "b.py"}
        issues, _ = _collect_completion_issues(ec)
        self.assertTrue(any("Declared but never written" in i for i in issues))


class IncompleteFinalizationTests(_ChecklistFixture):
    def test_all_validated_with_no_checklist_is_complete(self):
        # Unchanged behaviour for the majority of runs.
        self.assertFalse(needs_incomplete_finalization(_written({"a.py"}, tier="oracle")))

    def test_all_validated_but_checklist_open_is_incomplete(self):
        # The gap this closes: validating the two files you wrote is no evidence
        # about the three steps you never started.
        ec = _written({"a.py"}, tier="oracle")
        self.checklist(ec, ["- [x] one", "- [ ] two", "- [ ] three"])
        self.assertTrue(needs_incomplete_finalization(ec))

    def test_open_checklist_without_code_mutation_is_not_forced(self):
        ec = _ctx()
        self.checklist(ec, ["- [ ] think about it"])
        self.assertFalse(needs_incomplete_finalization(ec))

    def test_optional_steps_alone_do_not_force_it(self):
        ec = _written({"a.py"}, tier="oracle")
        self.checklist(ec, ["- [x] one", "- [ ] optional: extra benchmark"])
        self.assertFalse(needs_incomplete_finalization(ec))


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
        ec = _written({"a.py"}, tier="oracle")
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
            ran_out_of_steps=True,
        )
        self.assertTrue(out.startswith(HEADLINE_INCOMPLETE))
        self.assertIn("Not performed (you refused these", out)

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
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] one", "- [ ] two"])
        self.assertTrue(self._should(ec))

    def test_silent_without_a_checklist(self):
        self.assertFalse(self._should(_written({"a.py"}, tier="executed")))

    def test_silent_before_any_code_was_written(self):
        ec = _ctx()
        self.checklist(ec, ["- [ ] two"])
        self.assertFalse(self._should(ec))

    def test_silent_when_everything_is_ticked(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] one", "- [x] two"])
        self.assertFalse(self._should(ec))

    def test_optional_steps_alone_do_not_fire_it(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] one", "- [ ] (optional) extra"])
        self.assertFalse(self._should(ec))

    def test_capped_at_one(self):
        from mimir.client.config.constants import NUDGE_MAX_UNFINISHED_PLAN
        ec = _written({"a.py"}, tier="executed")
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
    """The workspace root must be unambiguous in the injected orientation.

    Regression, observed twice. A run told to create files "outside of the codes
    directory" created them in the workspace root — which *is* `codes` — and reported
    the constraint satisfied. The first fix added an absolute "Workspace root" line but
    left the tree rendered from the basename (`codes/ (6 dirs, 18 files)`), so the root
    still read as a *child* of the workspace and "the workspace root" still looked like
    a way out of it. The tree's own root line has to be absolute: that is where the
    model actually reads containment from.
    """

    def _snapshot(self):
        import tempfile
        from mimir.client.prompt.repo_baseline import build_repo_baseline_snapshot
        root = os.path.join(tempfile.mkdtemp(), "codes")
        os.makedirs(os.path.join(root, "mimir"))
        open(os.path.join(root, "README.md"), "w").close()
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        return root, build_repo_baseline_snapshot(root=root)

    def test_tree_root_line_is_absolute_not_a_bare_name(self):
        root, snap = self._snapshot()
        ctx = snap["context"]
        self.assertIn(f"{root}/ (", ctx)
        # The bare-basename form is what made the root look like a subdirectory.
        self.assertNotRegex(ctx, r"(?m)^codes/ \(")

    def test_absolute_root_is_stated_explicitly(self):
        root, snap = self._snapshot()
        self.assertIn(f"Workspace root (absolute): {root}", snap["context"])
        self.assertEqual(snap["root"], root)

    def test_prompt_points_at_the_root_for_building_absolute_paths(self):
        """The root is now a *prerequisite*, not a hint.

        Placement is enforced at the tool boundary (file tools reject relative
        paths), so the prompt no longer argues about how relative paths resolve —
        it only has to tell the model where to join from. The prose that tried to
        make the inference reliable is gone; see server_files._require_abs.
        """
        from mimir.client.prompt.system_prompt import build_system_content
        root, snap = self._snapshot()
        content = build_system_content(
            active_mode="agent", tool_owner={}, sensitive_tools=set(),
            platform_profile_summary="", repo_baseline_context=snap["context"],
            memory_context_file="", todo_file="", plan_todos=None,
        )
        self.assertIn(snap["context"], content)
        self.assertIn("File tools take absolute paths", content)
        # The superseded band-aids must not linger alongside the structural fix.
        self.assertNotIn("workspace root is NOT a way out of it", content)
        self.assertNotIn("Every relative path you give resolves", content)

    def test_ledger_does_not_restate_the_workspace_root(self):
        # Was a compensating disclosure for ambiguous relative paths. The model now
        # types the destination explicitly, so restating it is noise on every answer.
        ec = _written({"wave_solver_2d/solver.py"}, tier="syntax")
        out = _annotate_answer_with_changes("Created outside the codes directory.", ec)
        self.assertIn("`wave_solver_2d/solver.py` — validated: syntax", out)
        self.assertNotIn("workspace root", out)


if __name__ == "__main__":
    unittest.main()
