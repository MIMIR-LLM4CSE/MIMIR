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
    _collect_completion_issues,
    unchecked_checklist_items,
)
from mimir.client.query_engine.finalize import _annotate_answer_with_changes


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
        self.assertIn("[Verification ledger", out)
        self.assertIn("a.py (NOT validated)", out)

    def test_tier_is_reported_per_file(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="static"))
        self.assertIn("a.py (validated: static)", out)

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
        self.assertIn("parser.py (validated: oracle — red→green)", out)
        self.assertNotIn("Reported invariant", out)

        invariant = _written({"solver.py"}, tier="oracle")
        out = _annotate_answer_with_changes("Done.", invariant)
        self.assertIn("solver.py (validated: oracle — reported invariant)", out)
        # An invariant is presence-only evidence; the ledger must not let it pass for
        # a comparison against something sealed.
        self.assertIn("never its value", out)

    def test_declared_but_never_written_is_reported(self):
        ec = _written({"a.py"}, tier="executed")
        ec["declared_edit_set"] = {"a.py", "b.py"}
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("Declared but never written: b.py", out)

    def test_unchecked_steps_are_reported_with_a_preview(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] write solver", "- [ ] add convergence test",
                            "- [ ] document it"])
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("2 step(s) unchecked", out)
        self.assertIn("add convergence test", out)

    def test_optional_steps_are_counted_separately(self):
        ec = _written({"a.py"}, tier="executed")
        self.checklist(ec, ["- [x] write solver", "- [ ] (optional) convergence study"])
        out = _annotate_answer_with_changes("Done.", ec)
        self.assertIn("1 optional step(s) not done", out)
        self.assertNotIn("step(s) unchecked", out)

    def test_no_checklist_yields_no_checklist_lines(self):
        out = _annotate_answer_with_changes("Done.", _written({"a.py"}, tier="oracle"))
        self.assertNotIn("Checklist", out)

    def test_ledger_never_replaces_the_model_answer(self):
        out = _annotate_answer_with_changes("Prose.", _written({"a.py"}, tier="oracle"))
        self.assertTrue(out.startswith("Prose."))


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
        self.assertIn("Files written: wave_solver_2d/solver.py", out)
        self.assertNotIn("workspace root", out)


if __name__ == "__main__":
    unittest.main()
