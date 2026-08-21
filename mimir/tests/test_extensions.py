"""Coverage for the pluggable policy/nudge extension seam (``client.extensions``).

Asserts that application-registered PolicyChecks block calls at their stage, that
NudgeRules fire / are tier-gated / are toggle-suppressed, that an empty registry leaves
core behaviour byte-identical, that registration is idempotent by name, and that the
directory-scan loader imports a pack module. The module-global registries are cleared in
setUp/tearDown so nothing leaks between tests or into other test modules.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import types
import unittest

import mimir.client.guardrails.policy.engine as policy_manager_module
from mimir.client.extensions import (
    NudgeRule,
    PolicyCheck,
    PostToolRule,
    register_nudge,
    register_policy_check,
    register_post_tool,
)
from mimir.client.guardrails.policy.plugins import PolicyRegistry
from mimir.client.guardrails.nudges.engine import maybe_append_nudge
from mimir.client.guardrails.nudges.plugins import NudgeRegistry
from mimir.client.tool_execution.executor import run_post_tool_annotations
from mimir.client.tool_execution.plugins import PostToolRegistry
from mimir.client.extensions.plugins import load_plugins
from mimir.tests.test_policy_manager import _FakeAgent


def _block_payload(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


class _NudgeAgent:
    """Minimal agent surface consumed by the nudge dispatcher."""

    def __init__(self, enforcement: str = "strict", disabled: set | None = None) -> None:
        self.enforcement = enforcement
        self.model = "dummy"
        self.tool_caps: dict = {}
        self.disabled_nudges = disabled or set()


class ExtensionRegistryResetMixin(unittest.TestCase):
    def setUp(self) -> None:
        PolicyRegistry.clear()
        NudgeRegistry.clear()
        PostToolRegistry.clear()

    def tearDown(self) -> None:
        PolicyRegistry.clear()
        NudgeRegistry.clear()
        PostToolRegistry.clear()


class PolicyPluginTests(ExtensionRegistryResetMixin):
    def test_pre_approval_check_blocks_matching_call(self) -> None:
        register_policy_check(PolicyCheck(
            name="test_block",
            stage="pre_approval",
            check=lambda agent, tool, args, ec: (
                _block_payload("nope") if args.get("boom") else None
            ),
        ))
        agent = _FakeAgent()
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent, tool_name="read_file_lines", arguments={"boom": True}, execution_context={},
        )
        self.assertIsNotNone(result.violation)
        payload = json.loads(result.violation)
        self.assertEqual(payload["policy_stage"], "test_block")

    def test_pre_mutation_stage_runs(self) -> None:
        register_policy_check(PolicyCheck(
            name="veto_all",
            stage="pre_mutation",
            check=lambda agent, tool, args, ec: _block_payload("vetoed"),
        ))
        agent = _FakeAgent()
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent, tool_name="read_file_lines", arguments={}, execution_context={},
        )
        self.assertIsNotNone(result.violation)
        self.assertEqual(json.loads(result.violation)["policy_stage"], "veto_all")

    def test_non_matching_check_allows_call(self) -> None:
        register_policy_check(PolicyCheck(
            name="never",
            check=lambda agent, tool, args, ec: None,
        ))
        agent = _FakeAgent()
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent, tool_name="read_file_lines", arguments={}, execution_context={},
        )
        self.assertIsNone(result.violation)

    def test_empty_registry_core_unchanged(self) -> None:
        # No registered checks: a benign read passes exactly as core does today.
        agent = _FakeAgent()
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent, tool_name="read_file_lines", arguments={}, execution_context={},
        )
        self.assertIsNone(result.violation)

    def test_raising_check_is_ignored_not_fatal(self) -> None:
        def _boom(agent, tool, args, ec):
            raise RuntimeError("bad pack")

        register_policy_check(PolicyCheck(name="raiser", check=_boom))
        agent = _FakeAgent()
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent, tool_name="read_file_lines", arguments={}, execution_context={},
        )
        self.assertIsNone(result.violation)  # a raising check is "no opinion"

    def test_register_is_idempotent_by_name(self) -> None:
        register_policy_check(PolicyCheck(name="dup", check=lambda a, t, ar, e: None))
        register_policy_check(PolicyCheck(name="dup", check=lambda a, t, ar, e: _block_payload("x")))
        self.assertEqual(PolicyRegistry.names(), ["dup"])
        self.assertEqual(len(PolicyRegistry.active_checks()), 1)

    def test_invalid_stage_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolicyCheck(name="bad", check=lambda a, t, ar, e: None, stage="whenever")


class NudgePluginTests(ExtensionRegistryResetMixin):
    def _fire(self, agent, rule_kwargs) -> list[dict]:
        register_nudge(NudgeRule(
            name="authz",
            predicate=lambda ag, q, m, ec: True,
            render=lambda ag, ec: "Confirm this action is authorized.",
            **rule_kwargs,
        ))
        messages: list[dict] = []
        maybe_append_nudge(
            agent=agent, query="say hi", active_mode="agent",
            execution_context={}, messages=messages,
        )
        return messages

    def test_guidance_nudge_fires(self) -> None:
        messages = self._fire(_NudgeAgent(enforcement="strict"), {"layer": "guidance"})
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("authorized", messages[0]["content"])

    def test_disabled_nudge_suppressed(self) -> None:
        messages = self._fire(_NudgeAgent(enforcement="strict", disabled={"authz"}), {"layer": "guidance"})
        self.assertEqual(messages, [])

    def test_guidance_tier_gating(self) -> None:
        # Rule limited to strict: fires under strict, silent under light.
        strict_msgs = self._fire(
            _NudgeAgent(enforcement="strict"),
            {"layer": "guidance", "tiers": frozenset({("strict", "agent")})},
        )
        self.assertEqual(len(strict_msgs), 1)

        NudgeRegistry.clear()
        light_msgs = self._fire(
            _NudgeAgent(enforcement="light"),
            {"layer": "guidance", "tiers": frozenset({("strict", "agent")})},
        )
        self.assertEqual(light_msgs, [])

    def test_guidance_skipped_when_enforcement_off(self) -> None:
        messages = self._fire(_NudgeAgent(enforcement="off"), {"layer": "guidance"})
        self.assertEqual(messages, [])

    def test_verification_nudge_ignores_enforcement(self) -> None:
        # Verification-layer rules run even at enforcement "off".
        messages = self._fire(_NudgeAgent(enforcement="off"), {"layer": "verification"})
        self.assertEqual(len(messages), 1)

    def test_empty_registry_no_custom_nudge(self) -> None:
        messages: list[dict] = []
        fired = maybe_append_nudge(
            agent=_NudgeAgent(), query="say hi", active_mode="agent",
            execution_context={}, messages=messages,
        )
        self.assertFalse(fired)
        self.assertEqual(messages, [])

    def test_per_query_cap(self) -> None:
        register_nudge(NudgeRule(
            name="spammy", layer="verification",
            predicate=lambda ag, q, m, ec: True,
            render=lambda ag, ec: "again",
        ))
        agent = _NudgeAgent()
        ec: dict = {}
        for _ in range(6):
            maybe_append_nudge(agent=agent, query="hi", active_mode="agent",
                               execution_context=ec, messages=[])
        # Capped at CUSTOM_NUDGE_MAX_PER_QUERY (3) via the shared nudge_counts.
        self.assertEqual(ec["nudge_counts"]["spammy"], 3)


class PostToolPluginTests(ExtensionRegistryResetMixin):
    """The seam between a call and the turn's end: react to what a tool produced."""

    def _agent(self):
        return types.SimpleNamespace(
            tool_caps={},
            _normalize_workspace_path=lambda p: p or "",
        )

    def _annotate(self, result: str = "{}") -> str:
        return asyncio.run(run_post_tool_annotations(
            self._agent(), "any_tool", {}, result, {},
        ))

    def test_an_empty_registry_annotates_nothing(self) -> None:
        self.assertEqual(self._annotate(), "")

    def test_a_sync_hook_appends_its_text(self) -> None:
        register_post_tool(PostToolRule(
            name="note", run=lambda ag, t, ar, res, ec: "\n\nNOTE: seen",
        ))
        self.assertIn("NOTE: seen", self._annotate())

    def test_an_async_hook_is_awaited(self) -> None:
        async def _run(agent, tool_name, arguments, result, execution_context):
            return "\n\nASYNC: ok"

        register_post_tool(PostToolRule(name="async_note", run=_run))
        self.assertIn("ASYNC: ok", self._annotate())

    def test_the_hook_sees_the_tool_result(self) -> None:
        # What the call produced is the whole point of this seam.
        register_post_tool(PostToolRule(
            name="echo",
            run=lambda ag, t, ar, res, ec: "\n\nRED" if "error" in res else "",
        ))
        self.assertIn("RED", self._annotate('{"status": "error"}'))
        self.assertEqual(self._annotate('{"status": "ok"}'), "")

    def test_a_raising_hook_is_skipped_not_fatal(self) -> None:
        def _boom(agent, tool_name, arguments, result, execution_context):
            raise RuntimeError("bad pack")

        register_post_tool(PostToolRule(name="boom", order=10, run=_boom))
        register_post_tool(PostToolRule(
            name="survivor", order=20, run=lambda ag, t, ar, res, ec: "\n\nSTILL HERE",
        ))
        self.assertIn("STILL HERE", self._annotate())

    def test_hooks_run_in_declared_order(self) -> None:
        register_post_tool(PostToolRule(name="b", order=20, run=lambda *a: "second"))
        register_post_tool(PostToolRule(name="a", order=10, run=lambda *a: "first"))
        self.assertEqual(self._annotate(), "firstsecond")

    def test_a_hook_can_write_to_the_blackboard(self) -> None:
        # It cannot block — the call already happened — but the turn-end gates read
        # what it records, which is how a machine-observed failure stops a conclusion.
        def _record(agent, tool_name, arguments, result, execution_context):
            execution_context.setdefault("dirty_written_files", set()).add("solver.py")
            return ""

        register_post_tool(PostToolRule(name="record", run=_record))
        ec: dict = {}
        asyncio.run(run_post_tool_annotations(self._agent(), "any_tool", {}, "{}", ec))
        self.assertEqual(ec["dirty_written_files"], {"solver.py"})


class LoaderTests(ExtensionRegistryResetMixin):
    def test_directory_scan_imports_pack(self) -> None:
        pack = (
            "from mimir.client.extensions import PolicyCheck, NudgeRule, "
            "register_policy_check, register_nudge\n"
            "register_policy_check(PolicyCheck(name='scanned_policy', "
            "check=lambda a, t, ar, e: None))\n"
            "register_nudge(NudgeRule(name='scanned_nudge', layer='guidance', "
            "predicate=lambda a, q, m, e: False, render=lambda a, e: 'x'))\n"
        )
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "mypack.py"), "w", encoding="utf-8") as f:
                f.write(pack)
            # A private/helper module is skipped.
            with open(os.path.join(d, "_helper.py"), "w", encoding="utf-8") as f:
                f.write("x = 1\n")
            loaded = load_plugins(d)

        self.assertEqual(loaded, 1)
        self.assertIn("scanned_policy", PolicyRegistry.names())
        self.assertIn("scanned_nudge", NudgeRegistry.names())

    def test_missing_dir_is_noop(self) -> None:
        self.assertEqual(load_plugins("/nonexistent/path/plugins"), 0)

    def test_bad_pack_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "broken.py"), "w", encoding="utf-8") as f:
                f.write("raise RuntimeError('boom at import')\n")
            self.assertEqual(load_plugins(d), 0)  # skipped, not fatal


if __name__ == "__main__":
    unittest.main()
