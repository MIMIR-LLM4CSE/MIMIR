"""Workstream C: prompt-prefix stability for vLLM prefix caching.

Asserts the checklist pin is a *transient* tail message that never mutates the
static system message (messages[0]) and nets to zero after a step, and that the
per-query tool list is identical regardless of evolving discovery state.
"""
import os
import tempfile
import unittest

from mimir.client.query_engine import agent_loop as m
from mimir.client.query_engine import toollist as pe
from mimir.tests._golden_caps import build_declared_registry


def _ctx(lines=("- [ ] two",)):
    """An execution context whose only pinnable state is a live checklist."""
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "todo_list.md")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"todo_file_path": fp}


class PinTransienceTest(unittest.TestCase):
    def test_tail_user_pin_nets_to_zero_and_leaves_system_untouched(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        original = [dict(x) for x in messages]
        token = m._inject_pin(messages, _ctx())
        self.assertIsNotNone(token)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertTrue(messages[-1]["content"].startswith(m._PIN_MARKER))
        self.assertEqual(messages[0], original[0])  # static system prompt untouched
        m._remove_pin(messages, token)
        self.assertEqual(messages, original)  # net zero

    def test_the_pin_is_never_a_tail_system_message(self):
        # A template is free to fold a tail system message into the preceding turn,
        # and the DeepSeek one drops the generation prompt with it — the model is then
        # asked to continue the pin's text instead of answering. A tail user turn always
        # renders with the assistant marker after it; strict-alternation templates are
        # served by the backend's consecutive-user merge.
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        m._inject_pin(messages, _ctx())
        self.assertNotEqual(messages[-1]["role"], "system")

    def test_empty_pin_is_noop(self):
        messages = [{"role": "system", "content": "SYS"}]
        token = m._inject_pin(messages, {})  # sparse ctx -> empty pin
        self.assertIsNone(token)
        self.assertEqual(len(messages), 1)


class ChecklistIsPinnableAloneTest(unittest.TestCase):
    """A live checklist must reach the pin — it is the only thing the pin carries.

    The copy in messages[0] is a build-time snapshot rebuilt only on a mode/thinking
    switch, so the pin is the sole live channel for it, and every mechanism that
    holds the model to its checklist depends on the model being able to see it.
    """

    def _ctx_with_checklist(self, lines):
        ctx = _ctx(lines)
        fp = ctx["todo_file_path"]
        self.addCleanup(lambda: (os.remove(fp), os.rmdir(os.path.dirname(fp))))
        return ctx

    def test_checklist_alone_produces_a_pin(self):
        from mimir.client.prompt.system_prompt import build_checklist_pin_block
        ctx = self._ctx_with_checklist(["- [x] one", "- [ ] two"])
        pin = build_checklist_pin_block(ctx)
        self.assertTrue(pin)
        self.assertIn("Task checklist (1 pending)", pin)
        self.assertIn("[ ] two", pin)

    def test_discovery_evidence_alone_produces_no_pin(self):
        from mimir.client.prompt.system_prompt import build_checklist_pin_block
        self.assertEqual(
            build_checklist_pin_block({"read_files": {"a.py"}, "existing_paths": {"a.py"}}),
            "",
        )

    def test_no_checklist_is_still_empty(self):
        from mimir.client.prompt.system_prompt import build_checklist_pin_block
        self.assertEqual(build_checklist_pin_block({}), "")

    def test_pin_with_only_a_checklist_still_nets_to_zero(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
        original = [dict(x) for x in messages]
        token = m._inject_pin(messages, self._ctx_with_checklist(["- [ ] two"]))
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
