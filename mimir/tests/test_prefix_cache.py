"""Workstream C: prompt-prefix stability for vLLM prefix caching.

Asserts the discovery pin is a *transient* tail message that never mutates the
static system message (messages[0]) and nets to zero after a step, and that the
per-query tool list is identical regardless of evolving discovery state.
"""
import unittest

from mimir.client.query_engine import agent_loop as m
from mimir.client.query_engine import toollist as pe
from mimir.tests._golden_caps import build_declared_registry


def _ctx():
    return {"read_files": {"a.py"}, "existing_paths": {"a.py"}}


class PinTransienceTest(unittest.TestCase):
    def test_tail_system_pin_nets_to_zero_and_leaves_system_untouched(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        original = [dict(x) for x in messages]
        token = m._inject_pin(messages, _ctx(), "system")
        self.assertIsNotNone(token)
        self.assertEqual(messages[-1]["role"], "system")
        self.assertTrue(messages[-1]["content"].startswith(m._PIN_MARKER))
        self.assertEqual(messages[0], original[0])  # static system prompt untouched
        m._remove_pin(messages, token)
        self.assertEqual(messages, original)  # net zero

    def test_append_user_mode_folds_and_restores(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        original = [dict(x) for x in messages]
        token = m._inject_pin(messages, _ctx(), "append_user")
        self.assertEqual(len(messages), 2)  # no new message
        self.assertTrue(messages[1]["content"].startswith("q"))
        self.assertIn(m._PIN_MARKER, messages[1]["content"])
        m._remove_pin(messages, token)
        self.assertEqual(messages, original)

    def test_empty_pin_is_noop(self):
        messages = [{"role": "system", "content": "SYS"}]
        token = m._inject_pin(messages, {}, "system")  # sparse ctx -> empty pin
        self.assertIsNone(token)
        self.assertEqual(len(messages), 1)


class ChecklistIsPinnableAloneTest(unittest.TestCase):
    """A live checklist must reach the pin even as the only evidence.

    Regression: the checklist was appended *after* an early ``if not lines: return
    ""``, so a run with no reads, writes or searches yet — the state right after a
    plan is approved — silently lost it. The copy in messages[0] is a build-time
    snapshot rebuilt only on a mode/thinking switch, so that left no live channel
    at all, and every mechanism that holds the model to its checklist depends on
    the model being able to see it.
    """

    def _ctx_with_checklist(self, lines):
        import os
        import tempfile
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "todo_list.md")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self.addCleanup(lambda: (os.remove(fp), os.rmdir(d)))
        return {"todo_file_path": fp}

    def test_checklist_alone_produces_a_pin(self):
        from mimir.client.prompt.system_prompt import build_discovery_pin_block
        ctx = self._ctx_with_checklist(["- [x] one", "- [ ] two"])
        pin = build_discovery_pin_block(ctx)
        self.assertTrue(pin)
        self.assertIn("Task checklist (1 pending)", pin)
        self.assertIn("[ ] two", pin)

    def test_checklist_still_appended_alongside_other_evidence(self):
        from mimir.client.prompt.system_prompt import build_discovery_pin_block
        ctx = self._ctx_with_checklist(["- [ ] two"])
        ctx.update(_ctx())
        pin = build_discovery_pin_block(ctx)
        self.assertIn("a.py", pin)
        self.assertIn("Task checklist", pin)

    def test_no_checklist_and_no_evidence_is_still_empty(self):
        from mimir.client.prompt.system_prompt import build_discovery_pin_block
        self.assertEqual(build_discovery_pin_block({}), "")

    def test_pin_with_only_a_checklist_still_nets_to_zero(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        original = [dict(x) for x in messages]
        token = m._inject_pin(messages, self._ctx_with_checklist(["- [ ] two"]), "system")
        self.assertIsNotNone(token)
        self.assertEqual(messages[0], original[0])
        m._remove_pin(messages, token)
        self.assertEqual(messages, original)


class ToolListStabilityTest(unittest.TestCase):
    def test_tool_list_is_identical_across_discovery_state(self):
        reg = build_declared_registry()
        tools = [{"function": {"name": n}} for n in
                 ("read_file_lines", "replace_in_file", "grep", "find_definition")]
        before = pe.tools_for_context(query="update the parser module",
                                      execution_context={}, tools=tools,
                                      tool_caps=reg, max_tools=40)
        after = pe.tools_for_context(query="update the parser module",
                                     execution_context={"searched": True, "read_files": {"x.py"}},
                                     tools=tools, tool_caps=reg, max_tools=40)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
