"""Workstream C: where the task checklist lives, and prompt-prefix stability.

Appending the checklist as a transient tail message before every model call keeps the
prefix byte-stable, but puts a block of *state* in the last position before the generation
prompt — which is what empties turns on a served model (37/108 draws with it, 0/84 without;
0/40 once the same text sits in messages[0]). These tests hold the checklist to messages[0]
and hold the rebuild to the moments the checklist actually changed.
"""
from mimir.tests.support.context import build_execution_context
import asyncio
import os
import tempfile
import unittest

from mimir.client.prompt.system_prompt import build_system_content
from mimir.client.query_engine import agent_loop as m
from mimir.client.guardrails import toollist as pe
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


class ChecklistPlacementTests(unittest.TestCase):
    """messages[0], and nowhere else — least of all the last position."""

    def _todo(self, body: str) -> str:
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "todo_list.md")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(body)
        self.addCleanup(lambda: (os.path.exists(fp) and os.remove(fp), os.rmdir(d)))
        return fp

    def _sync(self, agent, messages, ctx, system_content="SYS"):
        return asyncio.run(
            m._sync_checklist(agent, messages, "agent", system_content, ctx)
        )

    def _bump(self, fp: str, seconds: int = 10) -> None:
        """Move the mtime forward explicitly: two writes in the same test can land on
        one timestamp, and what is under test is the gate, not the clock."""
        st = os.stat(fp)
        os.utime(fp, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))

    def test_the_checklist_lands_in_messages_0(self):
        fp = self._todo("- [x] one\n- [ ] two\n")
        agent = _FakeAgent(fp)
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        out = self._sync(agent, messages, build_execution_context(todo_file_path=fp))
        self.assertIn("Task checklist (1 pending", messages[0]["content"])
        self.assertIn("[ ] two", messages[0]["content"])
        self.assertEqual(out, messages[0]["content"])

    def test_it_appends_nothing_and_reaches_no_other_message(self):
        # The regression this whole change exists to prevent: a block of state in the
        # last position before the generation prompt.
        fp = self._todo("- [ ] two\n")
        agent = _FakeAgent(fp)
        messages = [{"role": "system", "content": "SYS"},
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "tool", "content": "t"}]
        self._sync(agent, messages, build_execution_context(todo_file_path=fp))
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1]["role"], "tool")
        for msg in messages[1:]:
            self.assertNotIn("Task checklist", str(msg.get("content", "")))

    def test_an_unchanged_file_is_not_rebuilt(self):
        fp = self._todo("- [ ] two\n")
        agent = _FakeAgent(fp)
        ctx = build_execution_context(todo_file_path=fp)
        messages = [{"role": "system", "content": "SYS"}]
        self._sync(agent, messages, ctx)
        first, builds = messages[0]["content"], agent.builds
        for _ in range(3):
            self._sync(agent, messages, ctx)
        self.assertEqual(messages[0]["content"], first)  # byte-identical prefix
        self.assertEqual(agent.builds, builds)  # the mtime gate held

    def test_a_touch_that_changes_nothing_costs_nothing(self):
        # The todo server re-saves the file even when the text is unchanged (ticking an
        # item already ticked). The mtime moves; the prompt must not.
        fp = self._todo("- [x] one\n")
        agent = _FakeAgent(fp)
        ctx = build_execution_context(todo_file_path=fp)
        messages = [{"role": "system", "content": "SYS"}]
        self._sync(agent, messages, ctx)
        first = messages[0]["content"]
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("- [x] one\n")
        self._bump(fp)
        self._sync(agent, messages, ctx)
        self.assertIs(messages[0]["content"], first)  # not even reassigned

    def test_a_real_tick_reaches_the_prompt(self):
        fp = self._todo("- [ ] one\n- [ ] two\n")
        agent = _FakeAgent(fp)
        ctx = build_execution_context(todo_file_path=fp)
        messages = [{"role": "system", "content": "SYS"}]
        self._sync(agent, messages, ctx)
        self.assertIn("Task checklist (2 pending", messages[0]["content"])
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("- [x] one\n- [ ] two\n")
        self._bump(fp)
        self._sync(agent, messages, ctx)
        self.assertIn("Task checklist (1 pending", messages[0]["content"])

    def test_the_skill_block_survives_a_rebuild(self):
        # The skill context is folded into messages[0], not appended as a second system
        # message, so a bare rebuild would silently drop it.
        fp = self._todo("- [ ] one\n")
        agent = _FakeAgent(fp)
        ctx = build_execution_context(todo_file_path=fp, skill_suffix="\n\nSKILL CONTEXT (SUBORDINATE). xyz")
        messages = [{"role": "system", "content": "SYS"}]
        self._sync(agent, messages, ctx)
        self.assertTrue(messages[0]["content"].endswith("SKILL CONTEXT (SUBORDINATE). xyz"))
        self.assertIn("Task checklist", messages[0]["content"])

    def test_no_checklist_leaves_the_prompt_alone(self):
        messages = [{"role": "system", "content": "SYS"}]
        out = self._sync(_FakeAgent(""), messages, build_execution_context())
        self.assertEqual(messages, [{"role": "system", "content": "SYS"}])
        self.assertEqual(out, "SYS")

    def test_a_missing_todo_file_is_not_an_error(self):
        messages = [{"role": "system", "content": "SYS"}]
        ctx = build_execution_context(todo_file_path="/nonexistent/todo_list.md")
        out = self._sync(_FakeAgent("/nonexistent/todo_list.md"), messages, ctx)
        self.assertEqual(out, "SYS")
        self.assertEqual(messages[0]["content"], "SYS")


class RebuildRuleTests(unittest.TestCase):
    """``messages[0]`` is the mode's prompt plus the folded skill block at every rebuild: mode
    switch, thinking rung, plan→agent handoff, checklist refresh.
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
