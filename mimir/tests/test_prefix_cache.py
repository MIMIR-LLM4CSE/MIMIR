"""Where the task checklist lives, and prompt-prefix stability.

A block of *state* in the last position before the generation prompt empties turns on a
served model (37/108 draws with it, 0/84 without; 0/40 once the same text sits in
messages[0]), and rewriting messages[0] mid-turn voids the prompt prefix for every step
after. So the checklist is rendered into messages[0] as the turn starts, and within the turn
it comes back in the checklist tool's own reply, pinned.
"""
from mimir.tests.support.context import build_execution_context
import asyncio
import json
import os
import tempfile
import types
import unittest

from mimir.client.prompt.system_prompt import build_system_content
from mimir.client.guardrails.policy import toollist as pe
from mimir.tests.support.golden_caps import build_declared_registry
import mimir.client.query_engine.loop_base as loop_base_module


class _FakeAgent:
    """Builds a real system prompt from a real todo file, and counts the builds."""

    def __init__(self, todo_fp: str) -> None:
        self.todo_fp = todo_fp
        self.builds = 0

    async def build_system_content(self, active_mode: str) -> str:
        self.builds += 1
        return build_system_content(
            active_mode=active_mode, tool_owner={}, sensitive_tools=set(),
            todo_file=self.todo_fp,
        )


def _todo_file(test: unittest.TestCase, body: str) -> str:
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "todo_list.md")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write(body)
    test.addCleanup(lambda: (os.path.exists(fp) and os.remove(fp), os.rmdir(d)))
    return fp


class ChecklistWithinTurnTests(unittest.TestCase):
    """The checklist is rendered into messages[0] as the turn starts, and only then.

    Within the turn a change comes back in the checklist tool's own reply, whose latest copy
    is pinned, so the system message, and the prompt prefix with it, holds for every step.
    """

    def _system(self, fp: str) -> str:
        return build_system_content(
            active_mode="agent", tool_owner={}, sensitive_tools=set(), todo_file=fp,
        )

    def test_the_turn_starts_with_the_checklist_in_the_system_message(self):
        out = self._system(_todo_file(self, "- [x] one\n- [ ] two\n"))
        self.assertIn("Task checklist at the start of this turn (1 pending", out)
        self.assertIn("[ ] two", out)

    def test_the_reply_renders_the_list_as_the_system_message_does(self):
        # One rendering, so the model reads the same lines in both places.
        from mimir.servers.agent_state import server_todo
        items = [{"index": 0, "text": "one", "done": True},
                 {"index": 1, "text": "two", "done": False}]
        reply = server_todo._checklist_reply(items, item=items[0])
        self.assertEqual(reply["pending"], 1)
        self.assertIn(reply["checklist"], self._system(_todo_file(self, "- [x] one\n- [ ] two\n")))

    def _pin(self, ctx, call_id, reply, name="todo_update"):
        from mimir.client.query_engine import dispatch as dispatch_module
        agent = types.SimpleNamespace(tool_caps=build_declared_registry())
        dispatch_module._pin_checklist(agent, name, call_id, reply, ctx)

    _OK = json.dumps({"status": "ok", "pending": 1, "checklist": "[x] one\n[ ] two"})

    def test_each_reply_takes_the_pin_from_the_last(self):
        ctx = build_execution_context()
        self._pin(ctx, "c1", self._OK)
        self._pin(ctx, "c2", self._OK)
        self.assertEqual(ctx.bookkeeping.pinned_call_ids, {"c2"})
        self.assertEqual(ctx.plan.checklist_call_id, "c2")

    def test_a_failed_update_leaves_the_last_list_pinned(self):
        ctx = build_execution_context()
        self._pin(ctx, "c1", self._OK)
        self._pin(ctx, "c2", json.dumps({"status": "error", "error": "Index 9 out of range"}))
        self.assertEqual(ctx.bookkeeping.pinned_call_ids, {"c1"})

    def test_a_tool_that_does_not_carry_the_checklist_is_not_pinned(self):
        ctx = build_execution_context()
        self._pin(ctx, "c1", self._OK, name="read_file_lines")
        self.assertEqual(ctx.bookkeeping.pinned_call_ids, set())

    def test_the_checklist_pin_leaves_a_skill_pin_alone(self):
        ctx = build_execution_context(skills_loaded={"fix-bug": "s1"}, pinned_call_ids={"s1"})
        self._pin(ctx, "c1", self._OK)
        self._pin(ctx, "c2", self._OK)
        self.assertEqual(ctx.bookkeeping.pinned_call_ids, {"s1", "c2"})


class RebuildRuleTests(unittest.TestCase):
    """``messages[0]`` is the mode's prompt plus the folded skill block at every rebuild: mode
    switch, thinking rung, plan→agent handoff.
    """

    def _agent(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "todo_list.md")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("- [ ] one\n")
        self.addCleanup(lambda: (os.remove(fp), os.rmdir(d)))
        return _FakeAgent(fp)

    def test_the_helper_reapplies_the_skill_block(self):
        ctx = build_execution_context(skill_suffix="\n\nSKILL CONTEXT (SUBORDINATE). xyz")
        out = asyncio.run(loop_base_module.rebuild_system_content(self._agent(), "agent", ctx))
        self.assertTrue(out.endswith("SKILL CONTEXT (SUBORDINATE). xyz"))
        self.assertIn("Task checklist", out)

    def test_no_skill_block_adds_nothing(self):
        out = asyncio.run(loop_base_module.rebuild_system_content(self._agent(), "agent", build_execution_context()))
        self.assertNotIn("SKILL CONTEXT", out)

    def test_a_mode_switch_keeps_the_skill_block(self):
        # The site the plan→agent handoff and the /mode toggle both land on.
        ctx = build_execution_context(skill_suffix="\n\nSKILL CONTEXT (SUBORDINATE). xyz")
        messages = [{"role": "system", "content": "OLD"}]
        out = asyncio.run(loop_base_module._apply_mode_switch(
            self._agent(), messages, new_mode="agent", execution_context=ctx,
        ))
        self.assertTrue(messages[0]["content"].endswith("SKILL CONTEXT (SUBORDINATE). xyz"))
        self.assertEqual(out, messages[0]["content"])

    def test_a_thinking_rung_change_keeps_the_skill_block(self):
        from mimir.client.config.settings import THINKING_DEPTH_AUTO
        agent = self._agent()
        agent.thinking_depth = THINKING_DEPTH_AUTO
        ctx = build_execution_context(skill_suffix="\n\nSKILL CONTEXT (SUBORDINATE). xyz")
        messages = [{"role": "system", "content": "OLD"}]
        auto, out = asyncio.run(loop_base_module._sync_thinking_directive(
            agent, messages, "agent", False, "OLD", ctx,
        ))
        self.assertTrue(auto)
        self.assertTrue(messages[0]["content"].endswith("SKILL CONTEXT (SUBORDINATE). xyz"))
        self.assertEqual(out, messages[0]["content"])


class PulledSkillLeavesThePrefixAloneTests(unittest.TestCase):
    """A skill the model pulls mid-run stays a tool result in the tail, protected there, so
    the prompt prefix is kept; a /name skill is folded into ``messages[0]``.
    """

    def _agent(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "todo_list.md")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("- [ ] one\n")
        self.addCleanup(lambda: (os.remove(fp), os.rmdir(d)))
        return _FakeAgent(fp)

    def test_a_pull_does_not_touch_the_system_message_or_the_suffix(self):
        from mimir.client.query_engine import dispatch as dispatch_module
        agent = self._agent()
        agent.tool_caps = {}   # the only thing the pin helper reads off the agent
        ctx = build_execution_context(skills_loaded={"fix-bug": "pending"})
        messages = [{"role": "system", "content": "SYS"}]
        before = messages[0]["content"]

        dispatch_module._pin_loaded_skill(
            agent, "load_skill", {"name": "fix-bug"}, "call-1",
            '{"status": "ok", "instructions": "METHOD BODY"}', ctx,
        )

        # Not even reassigned: the same string object is still there.
        self.assertIs(messages[0]["content"], before)
        self.assertFalse(ctx.bookkeeping.skill_suffix)
        self.assertEqual(ctx.bookkeeping.pinned_call_ids, {"call-1"})

    def test_a_rebuild_after_a_pull_carries_no_skill_block(self):
        """Because there is nothing in the suffix to carry: the body is in the tail."""
        ctx = build_execution_context(skills_loaded={"fix-bug": "call-1"}, pinned_call_ids={"call-1"})
        out = asyncio.run(loop_base_module.rebuild_system_content(self._agent(), "agent", ctx))
        self.assertNotIn("SKILL CONTEXT", out)


class PlanModeChecklistTests(unittest.TestCase):
    """The plan-mode prompt carries no checklist section, so nothing to refresh there."""

    def test_the_checklist_is_agent_mode_only(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "todo_list.md")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("- [x] a step from the previous task\n")
        self.addCleanup(lambda: (os.remove(fp), os.rmdir(d)))
        kw = dict(tool_owner={}, sensitive_tools=set(), todo_file=fp)
        self.assertIn("Task checklist", build_system_content(active_mode="agent", **kw))
        # A second query asking for a plan must not be shown the finished checklist of
        # the first one as the plan of record.
        self.assertNotIn("Task checklist", build_system_content(active_mode="plan", **kw))


class ToolListStabilityTest(unittest.TestCase):
    """The tools payload is byte-identical whenever nothing the user controls moved, so it is
    cached once per session.
    """

    TOOLS = [{"function": {"name": n}} for n in
             ("read_file_lines", "replace_in_file", "grep", "find_definition")]

    def _build(self, ec, query="update the parser module"):
        return pe.tools_for_context(
            query=query, execution_context=ec,
            tools=self.TOOLS, tool_caps=build_declared_registry(),
        )

    def test_tool_list_is_identical_across_discovery_state(self):
        before = self._build({})
        after = self._build(build_execution_context(searched=True, read_files={"x.py"}))
        self.assertEqual(before, after)

    def test_tool_list_is_identical_across_different_queries(self):
        """Cross-*query* stability, which the domain pruning used to break by design.

        A session's second request about a different subject must reuse the first's
        tokenized tool block, not invalidate it.
        """
        self.assertEqual(
            self._build({}, query="optimise ce solveur"),
            self._build({}, query="write a poem about the sea"),
        )

    def test_the_whole_advertised_surface_is_passed_through(self):
        # Nothing is dropped: the guarantee the removal of the two filters bought.
        self.assertEqual(self._build({}), self.TOOLS)

    def test_serialization_is_byte_identical(self):
        # Equality of dicts is not enough for a cache that hashes tokens: the rendered
        # JSON must match too, which it does only if key order is preserved as well.
        import json
        self.assertEqual(
            json.dumps(self._build({})),
            json.dumps(self._build(build_execution_context(searched=True, read_files={"x.py"}))),
        )


if __name__ == "__main__":
    unittest.main()
