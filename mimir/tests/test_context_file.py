"""Unit coverage for the modular general-context loader.

Verifies the resolution order (env var → workspace file → built-in default), the
replace-the-doctrine-half semantics (the core half survives every override),
defensive fallbacks (missing/empty/unreadable), and that the dynamic sections still
layer on top of a loaded base.

Plain ``unittest`` to match the rest of the suite.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import mimir.client.prompt.system_prompt as cb
import mimir.client.extensions.system_prompt as sp_resolver
from mimir.client.config.constants import SYSTEM_PROMPT_ENV, SYSTEM_PROMPT_FILENAME


class ResolveContextFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        # Point the "workspace" dir at an isolated tmp dir and clear the env var so
        # tests never read the real repo's .mimir/ or a developer's override.
        p1 = patch.object(sp_resolver, "MIMIR_DIR", self.tmp)
        p1.start(); self.addCleanup(p1.stop)
        p2 = patch.dict(os.environ, {}, clear=False)
        p2.start(); self.addCleanup(p2.stop)
        os.environ.pop(SYSTEM_PROMPT_ENV, None)

    def _write(self, name: str, text: str) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _assert_doctrine_is(self, expected: str, resolved: str) -> None:
        """The override replaces the doctrine half; the core half is always appended."""
        self.assertEqual(resolved, expected + "\n\n" + cb._CORE_SYSTEM_CONTENT)

    def test_default_when_nothing_set(self) -> None:
        self.assertEqual(cb.build_base_system_content(), cb._DEFAULT_BASE_SYSTEM_CONTENT)

    def test_env_var_replaces_base(self) -> None:
        path = self._write("custom.md", "MY CUSTOM CONTEXT")
        os.environ[SYSTEM_PROMPT_ENV] = path
        self._assert_doctrine_is("MY CUSTOM CONTEXT", cb.build_base_system_content())

    def test_workspace_file_used_when_env_unset(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "WORKSPACE CONTEXT")
        self._assert_doctrine_is("WORKSPACE CONTEXT", cb.build_base_system_content())

    def test_env_var_takes_precedence_over_workspace_file(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "WORKSPACE CONTEXT")
        path = self._write("custom.md", "ENV CONTEXT")
        os.environ[SYSTEM_PROMPT_ENV] = path
        self._assert_doctrine_is("ENV CONTEXT", cb.build_base_system_content())

    def test_explicit_override_arg_wins(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "WORKSPACE CONTEXT")
        path = self._write("explicit.md", "EXPLICIT CONTEXT")
        os.environ[SYSTEM_PROMPT_ENV] = self._write("env.md", "ENV CONTEXT")
        self._assert_doctrine_is("EXPLICIT CONTEXT", cb.build_base_system_content(path))

    def test_missing_file_falls_back_to_default(self) -> None:
        os.environ[SYSTEM_PROMPT_ENV] = os.path.join(self.tmp, "does_not_exist.md")
        self.assertEqual(cb.build_base_system_content(), cb._DEFAULT_BASE_SYSTEM_CONTENT)

    def test_empty_file_falls_back_to_default(self) -> None:
        path = self._write("empty.md", "   \n  ")
        os.environ[SYSTEM_PROMPT_ENV] = path
        self.assertEqual(cb.build_base_system_content(), cb._DEFAULT_BASE_SYSTEM_CONTENT)

    def test_build_system_content_layers_sections_on_loaded_base(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "BASE")
        out = cb.build_system_content(
            active_mode="agent",
            tool_owner={},
            sensitive_tools=set(),
            memory_context_file=os.path.join(self.tmp, "no_memory.md"),
        )
        self.assertTrue(out.startswith("BASE"))
        # The dynamic memory-pointer section is still appended on top of the base.
        self.assertIn("Persistent memories are stored under:", out)


class CoreSurvivesOverrideTests(unittest.TestCase):
    """An application prompt replaces the doctrine half and nothing else.

    The failure this guards is silent: a ``.mimir/system_prompt.md`` is a persona
    document, so it never restates the non-negotiables or the tool contracts, and a
    whole-prompt replacement drops them without a word in the trace.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        p1 = patch.object(sp_resolver, "MIMIR_DIR", self._tmp.name)
        p1.start(); self.addCleanup(p1.stop)
        p2 = patch.dict(os.environ, {}, clear=False)
        p2.start(); self.addCleanup(p2.stop)
        os.environ.pop(SYSTEM_PROMPT_ENV, None)
        path = os.path.join(self._tmp.name, SYSTEM_PROMPT_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Identity: domain engineer\n\nYou work on a specific platform.")
        self.out = cb.build_base_system_content()

    def test_core_sections_are_still_present(self) -> None:
        for heading in (
            "## Non-negotiables", "## Latitude", "## Tool results",
            "## Discovery", "## Editing", "## Validation", "## Running code",
            "## Planning & todo",
        ):
            with self.subTest(section=heading):
                self.assertIn(heading, self.out)

    def test_doctrine_sections_are_gone(self) -> None:
        # "## Planning & todo" is deliberately absent from this list: it is core, not
        # doctrine, because the finalization blocker reads the checklist it describes.
        for heading in ("## Style", "## Scope", "## Workflow", "## Reasoning"):
            with self.subTest(section=heading):
                self.assertNotIn(heading, self.out)
        self.assertNotIn("Mathematical Intelligence", self.out)

    def test_override_opens_the_prompt(self) -> None:
        # Identity first, hard rules in the recency slot.
        self.assertTrue(self.out.startswith("# Identity: domain engineer"))


class DefaultBaseShapeTests(unittest.TestCase):
    """Guard the *shape* of the built-in prompt, not its wording.

    The prompt was once a single concatenated literal with zero newlines — a flat
    wall of prose whose middle mid-size open-weight models drop. These assertions
    exist so that regression cannot come back unnoticed, and so the token budget
    cannot creep back up.
    """

    def test_default_base_is_sectioned_markdown(self) -> None:
        base = cb._DEFAULT_BASE_SYSTEM_CONTENT
        self.assertGreaterEqual(base.count("\n## "), 8, "sections must survive as real markdown headings")
        self.assertGreaterEqual(base.count("\n"), 40, "instructions must be one per line, not one paragraph")

    def test_default_base_stays_within_budget(self) -> None:
        # ~1.7k tokens. The pre-refactor prompt was 11566 chars; this ceiling leaves
        # room to edit without leaving room to drift back to a wall of text.
        # Raised 9000 → 9200 for the math-delimiter line in ## Style: the prompt sat
        # at 8984, i.e. 16 chars of headroom, so the old ceiling admitted no new
        # guidance at all.
        # Raised 9200 → 10500 for TIER 1b (correctness vs executability) in
        # ## Validation. Same story: the prompt sat at 9160, 40 chars of headroom.
        # This one is worth the tokens — Tier 1a's four rungs were labelled
        # "CORRECTNESS" while testing only that code runs, and nothing anywhere told
        # the model the difference. The depth (per-domain technique) lives in the
        # write-tests skill, loaded on demand; only the principle and the
        # no-oracle-say-so escape hatch are permanent. Still under the 11566-char
        # wall-of-text prompt this replaced, and the structural guarantees below
        # (real headings, one instruction per line) are the actual protection
        # against regressing to prose — length alone never was.
        # Raised 10500 → 11000 for the denial-triage line in ## Non-negotiables: the
        # prompt sat at 10430. A refused approval carries one of three meanings (wrong
        # means / unnecessary step / stop), and reading it as "an error to retry" is
        # exactly the non-self-correcting failure this section exists for. The nudge
        # and tool-result copy say the same thing situationally, but both are
        # enforcement-gated or arrive only after the fact; this line is the carrier
        # that survives every level.
        # Raised 11000 → 12100 for the verdict rule in ## Validation: the prompt sat at
        # 10974. Exit 0 only says the program ran to the end, and nothing downstream can
        # read the output it produced — no parser generalises across plots, tables, logs
        # and physical units. So an executed check no longer validates a file on its own,
        # and the obligation to judge its output is stated here, where it reaches every
        # run. The three outcomes and what `unknown` obliges are load-bearing: without
        # them a model picks `pass` to close the loop, which is the failure this exists
        # to stop.
        # Raised 12100 → 12250 for the verdict-scope line in ## Validation: the prompt
        # sat at 11974. One unqualified `pass` used to settle every open run at once,
        # so a run the model had not read was credited by a statement about another —
        # the same over-crediting the verdict rule above exists to stop, one level up.
        # The rule only works if the model knows the bracketed form, and it is asked
        # for at the moment it would otherwise write the broad `pass`.
        # Raised 12250 → 12900 for two rules in ## Validation / ## Running code: the
        # prompt sat at 12780. The verdict rule above says what to do with a run's
        # output but never that one must exist, and nothing else in the loop asked for
        # an execution — a model could write, lint, and stop, with the whole judging
        # machinery unreachable because it is only ever entered by running something.
        # "Judging presupposes running" is the half that was missing, and it reaches
        # every run. The second line names the test directory as the place a keepable
        # test goes: the execution rule offers that route, and without it the
        # scratchpad rule sends the test to the wrong tree. The acquisition list for an
        # `unknown` is deliberately NOT here — it is situational, and the nudge that
        # fires on a standing `unknown` carries it in full.
        # Went the other way once: -567 chars by moving call mechanics (verdict
        # outcomes, new_text rules, sub-agent signature, todo call order, shell
        # gating) to the tool descriptions that receive those calls. The dividing
        # line is reach, not repetition — a tool description is only in context when
        # its tool is advertised, and `tools_for_context` prunes by domain, so the
        # prompt keeps every *when/whether* rule and the tools keep the *how*.
        # Raised 12900 → 13000 for the latitude rebalance: the prompt sat at 12780.
        # Net cost is 53 chars, because the additions — the workflow modes being
        # re-entrant, a checklist tracking progress rather than scripting it, and
        # declaring a real step dependency instead of implying a total order — were
        # paid for by deleting a sentence that restated the "NEVER fake validation"
        # non-negotiable almost word for word. What they buy is the three readings
        # this loop kept producing: that the phases run once in order, that every
        # checklist step is owed, and that build/run are owed wherever possible.
        # Raised 13000 → 13500: the two foundational context blocks (repo structure,
        # target platform) were removed, and with them the two prompt lines that
        # referred to them; what came back in their place is the pair of
        # non-negotiables naming the allocation and optimization-session gates, whose
        # rules previously existed only in a violation payload the model read after
        # being blocked. Base sat at 13004 against the old ceiling.
        self.assertLess(len(cb._DEFAULT_BASE_SYSTEM_CONTENT), 13500)

    def test_env_resolution_cascade_lives_in_the_nudges_not_the_prompt(self) -> None:
        # The 5-step cascade is covered by env_resolution/env_cleanup nudges, which
        # fire on an actual unresolved import; keeping a copy here cost ~18% of the
        # permanent prompt for a rare case.
        base = cb._DEFAULT_BASE_SYSTEM_CONTENT
        for marker in ("(1) DISCOVER", "(2) INSTALL", "(3) CREATE", "(4) CLEANUP"):
            self.assertNotIn(marker, base)
        # The one-line residual must remain: at "light" enforcement the env nudges
        # are filtered out, so the prompt is the only carrier of that guidance.
        self.assertIn("environment problem, not a code defect", base)


class NudgeCoverageTests(unittest.TestCase):
    """Every enforcement-gated obligation must also live in the system prompt.

    Guidance nudges are filtered by ``_GUIDANCE_BY_LEVEL_MODE``: at "light" only a
    subset fires, at "off" none do. The prompt is the only carrier that survives all
    three levels, so an obligation that exists *only* as a guidance nudge silently
    vanishes for exactly the strong models running at light/off. This test walks the
    real nudge table so a newly added guidance nudge fails here until its rule is
    stated in the prompt too.
    """

    # nudge category -> a phrase in the prompt carrying the same obligation.
    # `validation` is absent because it is no longer guidance: it is a verification
    # row that fires at every enforcement level, so the prompt is not its only carrier.
    _EXPECTED = {
        "discovery": "Grep first, read second",
        "state": "discover (gather evidence)",
        "doc": "Update documentation when a change affects",
        "todo": "record the concrete steps as a todo list",
        "blast_radius": "search all references and summarise the impact",
        "env_resolution": "environment problem, not a code defect",
        "env_cleanup": "reversible obligation",
    }
    # `creation` is deliberately exempt: it is an anti-dithering prod ("you have
    # enough context, start writing"), not an obligation. Its absence at light/off
    # is harmless — a model trusted at those levels does not dither.
    _EXEMPT = {"creation"}

    def test_every_guidance_nudge_has_a_prompt_counterpart(self) -> None:
        from mimir.client.guardrails.nudges.engine import _CORE_NUDGES

        base = cb._DEFAULT_BASE_SYSTEM_CONTENT
        guidance = {n.name for n in _CORE_NUDGES if n.layer == "guidance"}
        self.assertEqual(
            guidance - self._EXEMPT,
            set(self._EXPECTED),
            "a guidance nudge was added or renamed: state its rule in the system prompt "
            "and map it here, or justify an exemption",
        )
        for category, phrase in self._EXPECTED.items():
            with self.subTest(nudge=category):
                self.assertIn(phrase, base)


class CoreNudgeCoverageTests(unittest.TestCase):
    """Every verification-layer obligation must live in the un-overridable half.

    Stricter than NudgeCoverageTests above and for a different reason: verification
    nudges fire at every enforcement level, so an application prompt that dropped
    their rule would leave the loop demanding something the model was never told.
    """

    _EXPECTED = {
        "denial": "A refused approval is an instruction, not an error",
        "error_recovery": "Copy anchor text verbatim from your most recent read",
        "validation": "Every file you modify is checked before this run may conclude",
        "regression": "the project already has tests covering what you touched",
        "unexercised": "judging presupposes running",
        "unfinished_plan": "closed by saying so in your answer, not by ticking it",
    }
    # No exemption. `unfinished_plan` used to be exempt on the ground that the nudge
    # states both acceptable endings, so it asks for nothing told in advance. That
    # covered the nudge and not the blocker: `needs_incomplete_finalization` refuses to
    # conclude while a non-optional step is open, which is a contract about an artifact
    # — the checklist — that only ## Planning & todo describes. While that section sat
    # in the overridable doctrine half, an application prompt deleted it and the loop
    # then blocked on something the model was never told to keep.
    _EXEMPT: set[str] = set()

    def test_every_verification_nudge_has_a_core_counterpart(self) -> None:
        from mimir.client.guardrails.nudges.engine import _CORE_NUDGES

        core = cb._CORE_SYSTEM_CONTENT
        verification = {n.name for n in _CORE_NUDGES if n.layer == "verification"}
        self.assertEqual(
            verification - self._EXEMPT,
            set(self._EXPECTED),
            "a verification nudge was added or renamed: state its rule in "
            "_CORE_SYSTEM_CONTENT and map it here, or justify an exemption",
        )
        for category, phrase in self._EXPECTED.items():
            with self.subTest(nudge=category):
                self.assertIn(phrase, core)

    def test_the_prompt_names_the_third_ending(self) -> None:
        # Done, impossible, disproportionate. The third is the one a trim would drop
        # first, and dropping it puts back the "if it is possible, it is owed" reading.
        core = cb._CORE_SYSTEM_CONTENT
        self.assertIn("SIMPLY feasible", core)
        self.assertIn("out of proportion", core)
        self.assertIn("correct ending", core)

    def test_core_stays_within_budget(self) -> None:
        # The core is the incompressible part every application pays for, on top of
        # its own prompt. Its own ceiling so that growth shows up here, where it is
        # least affordable, rather than being absorbed by the whole-base budget.
        # Raised 10000 → 10500 when ## Planning & todo moved into core: the loop
        # blocks finalization on the checklist it describes, so leaving it in the
        # overridable half let an application prompt delete a rule the loop still
        # enforces. The move costs core ~700 chars and the base nothing.
        self.assertLess(len(cb._CORE_SYSTEM_CONTENT), 10500)


class SubAgentSectionGateTests(unittest.TestCase):
    """The sub-agent contract is injected only when a delegation capability is connected."""

    def _build(self, delegation: bool, mode: str = "agent") -> str:
        return cb.build_system_content(
            active_mode=mode,
            tool_owner={},
            sensitive_tools=set(),
            delegation_available=delegation,
        )

    def test_section_absent_without_the_capability(self) -> None:
        self.assertNotIn("## Sub-agents", self._build(False))

    def test_section_present_when_connected(self) -> None:
        out = self._build(True)
        self.assertIn("## Sub-agents", out)
        # The section carries the delegation policy, not the call signature: how to
        # call it is the tool's own description, which is in context whenever the
        # tool is.
        self.assertIn("Broad reconnaissance is delegated by default", out)
        self.assertNotIn("readonly=True", out)
        self.assertNotIn("spawn_agent", out)

    def test_section_asks_for_the_fan_out_in_one_response(self) -> None:
        """Issued one per turn, the children do not run in parallel — which is the point."""
        self.assertIn("SAME response", self._build(True))

    def test_readonly_modes_carry_the_clause_only_when_it_can_be_acted_on(self) -> None:
        for mode in ("plan", "ask"):
            with self.subTest(mode=mode):
                self.assertIn("This exploration parallelises", self._build(True, mode))
                self.assertNotIn("This exploration parallelises", self._build(False, mode))

    def test_section_is_not_baked_into_the_default_base(self) -> None:
        self.assertNotIn("## Sub-agents", cb._DEFAULT_BASE_SYSTEM_CONTENT)


if __name__ == "__main__":
    unittest.main()
