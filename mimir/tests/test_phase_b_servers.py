"""Faithfulness gate — the live classification comes entirely from the servers.

Shipped code carries no hardcoded tool-classification list: each server declares
its tools' capabilities via ``@mcp.tool(**tool_caps(...))`` and the client reads
them from the per-agent registry. This test parses every server's decorator **from
source via AST** (no ``mcp`` import, so it runs on the x86 build host), builds the
``{name: ToolCaps}`` registry exactly as ``connect_server`` would, and asserts it
reproduces the golden expected classification (``mimir/tests/_golden_caps.py``).

End-to-end ``meta`` serialization through FastMCP/``list_tools`` is covered by
``pytest mimir`` + a live agent start on the ARM client.
"""

import unittest

from mimir.client.context import capabilities as caps
from mimir.tests import _golden_caps as golden


class DeclaredClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = golden.build_declared_registry()

    def test_cap_sets_match_golden(self):
        for cap, expected in golden.GOLDEN.items():
            with self.subTest(cap=cap):
                self.assertEqual(caps.names_with_cap(cap, self.reg), expected)

    def test_path_args_match_golden(self):
        for tool, expected in golden.PATH_ARGS_BY_TOOL.items():
            with self.subTest(tool=tool):
                self.assertEqual(caps.path_args(tool, self.reg), expected)
        with_path = {n for n, c in self.reg.items() if c.arg_roles.get("path")}
        self.assertEqual(with_path, set(golden.PATH_ARGS_BY_TOOL))

    def test_fallbacks_match_golden(self):
        for tool, expected in golden.FALLBACK_TOOLS.items():
            with self.subTest(tool=tool):
                self.assertEqual(caps.fallbacks(tool, self.reg), expected)
        with_fb = {n for n, c in self.reg.items() if c.fallbacks}
        self.assertEqual(with_fb, set(golden.FALLBACK_TOOLS))

    def test_edit_sig_roles_match_golden(self):
        for tool, expected in golden.EDIT_SIG_ARGS.items():
            with self.subTest(tool=tool):
                self.assertEqual(caps.arg_role(tool, "edit_sig", self.reg), expected)
        with_sig = {n for n, c in self.reg.items() if c.arg_roles.get("edit_sig")}
        self.assertEqual(with_sig, set(golden.EDIT_SIG_ARGS))

    def test_plan_steps_roles_match_golden(self):
        for tool, expected in golden.PLAN_STEPS_ARGS.items():
            with self.subTest(tool=tool):
                self.assertEqual(caps.arg_role(tool, "plan_steps", self.reg), expected)
        with_steps = {n for n, c in self.reg.items() if c.arg_roles.get("plan_steps")}
        self.assertEqual(with_steps, set(golden.PLAN_STEPS_ARGS))

    def test_confirm_gate_roles_match_golden(self):
        for tool, expected in golden.CONFIRM_GATE_ARGS.items():
            with self.subTest(tool=tool):
                self.assertEqual(caps.arg_role(tool, "confirm_gate", self.reg), expected)
        with_gate = {n for n, c in self.reg.items() if c.arg_roles.get("confirm_gate")}
        self.assertEqual(with_gate, set(golden.CONFIRM_GATE_ARGS))

    def test_verdict_roles_match_golden(self):
        for role, expected_by_tool in (
            ("verdict", golden.VERDICT_ARGS),
            ("verdict_reason", golden.VERDICT_REASON_ARGS),
            ("verdict_scope", golden.VERDICT_SCOPE_ARGS),
        ):
            for tool, expected in expected_by_tool.items():
                with self.subTest(role=role, tool=tool):
                    self.assertEqual(caps.arg_role(tool, role, self.reg), expected)
            declared = {n for n, c in self.reg.items() if c.arg_roles.get(role)}
            self.assertEqual(declared, set(expected_by_tool))

    def test_scope_kinds_match_golden(self):
        for tool, kind in golden.SCOPE_KIND_BY_TOOL.items():
            with self.subTest(tool=tool):
                spec = caps.scope_spec(tool, self.reg)
                self.assertIsNotNone(spec, f"{tool} declared no scope spec")
                self.assertEqual(spec.get("kind"), kind)
        with_scope = {n for n, c in self.reg.items() if c.scope}
        self.assertEqual(with_scope, set(golden.SCOPE_KIND_BY_TOOL))

    def test_preview_kinds_match_golden(self):
        for tool, kind in golden.PREVIEW_KIND_BY_TOOL.items():
            with self.subTest(tool=tool):
                spec = caps.preview_spec(tool, self.reg)
                self.assertIsNotNone(spec, f"{tool} declared no preview spec")
                self.assertEqual(spec.get("kind"), kind)
        with_preview = {n for n, c in self.reg.items() if c.preview}
        self.assertEqual(with_preview, set(golden.PREVIEW_KIND_BY_TOOL))

    def test_risk_notes_declared(self):
        with_note = {n for n, c in self.reg.items() if c.risk_note}
        self.assertEqual(with_note, golden.RISK_NOTE_TOOLS)

    def test_every_golden_tool_is_declared(self):
        """Every tool that should be classified actually self-declares."""
        expected = set().union(*golden.GOLDEN.values()) | set(golden.PATH_ARGS_BY_TOOL)
        missing = sorted(expected - set(self.reg))
        self.assertEqual(missing, [], f"golden-classified tools not declared by a server: {missing}")


if __name__ == "__main__":
    unittest.main()
