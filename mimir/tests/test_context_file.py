"""Unit coverage for the modular general-context loader.

Verifies the resolution order (env var → workspace file → built-in default), the
replace-not-append semantics, defensive fallbacks (missing/empty/unreadable), and
that the dynamic sections still layer on top of a loaded base.

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

    def test_default_when_nothing_set(self) -> None:
        self.assertEqual(cb.build_base_system_content(), cb._DEFAULT_BASE_SYSTEM_CONTENT)

    def test_env_var_replaces_base(self) -> None:
        path = self._write("custom.md", "MY CUSTOM CONTEXT")
        os.environ[SYSTEM_PROMPT_ENV] = path
        self.assertEqual(cb.build_base_system_content(), "MY CUSTOM CONTEXT")

    def test_workspace_file_used_when_env_unset(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "WORKSPACE CONTEXT")
        self.assertEqual(cb.build_base_system_content(), "WORKSPACE CONTEXT")

    def test_env_var_takes_precedence_over_workspace_file(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "WORKSPACE CONTEXT")
        path = self._write("custom.md", "ENV CONTEXT")
        os.environ[SYSTEM_PROMPT_ENV] = path
        self.assertEqual(cb.build_base_system_content(), "ENV CONTEXT")

    def test_explicit_override_arg_wins(self) -> None:
        self._write(SYSTEM_PROMPT_FILENAME, "WORKSPACE CONTEXT")
        path = self._write("explicit.md", "EXPLICIT CONTEXT")
        os.environ[SYSTEM_PROMPT_ENV] = self._write("env.md", "ENV CONTEXT")
        self.assertEqual(cb.build_base_system_content(path), "EXPLICIT CONTEXT")

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
        self.assertLess(len(cb._DEFAULT_BASE_SYSTEM_CONTENT), 11000)

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
    _EXPECTED = {
        "validation": "treat validation as a primary objective",
        "discovery": "Grep first, read second",
        "state": "discover (gather evidence)",
        "doc": "Update documentation when a change affects",
        "todo": "record the concrete ordered steps as a todo list",
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


class SubAgentSectionGateTests(unittest.TestCase):
    """The sub-agent contract is injected only when a sub-agent tool is connected."""

    def _build(self, tool_owner: dict[str, str]) -> str:
        return cb.build_system_content(
            active_mode="agent",
            tool_owner=tool_owner,
            sensitive_tools=set(),
        )

    def test_section_absent_without_the_capability(self) -> None:
        self.assertNotIn("## Sub-agents", self._build({}))

    def test_section_present_when_connected(self) -> None:
        out = self._build({"spawn_agent": "agent"})
        self.assertIn("## Sub-agents", out)
        self.assertIn("readonly=True", out)

    def test_section_is_not_baked_into_the_default_base(self) -> None:
        self.assertNotIn("## Sub-agents", cb._DEFAULT_BASE_SYSTEM_CONTENT)


if __name__ == "__main__":
    unittest.main()
