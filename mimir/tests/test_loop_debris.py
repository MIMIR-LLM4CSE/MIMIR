"""The loop stops leaving its own machinery in the conversation.

Session ``b2fd38f1`` ended unable to produce anything, and its saved history showed
why it could not recover: 21 copies of one reminder sentence, 17 duplicate workflow
reminders, 9 copies of the same skill block and 9 empty assistant turns — 62 messages
and ~10% of the prompt, all of it produced by the loop reacting to its own failures.
The first empty turn arrived on a perfectly ordinary prompt; everything after it was
the loop's answer to that turn making the next one worse.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from mimir.client.config.constants import AGENT_EMPTY_TURN_RETRIES
from mimir.client.guardrails.nudges import drop_transient_reminders, inject_reminder
from mimir.client.query_engine import agent_loop as agent_loop_module
from mimir.client.query_engine import finalize as finalize_module
from mimir.client.query_engine import streaming as streaming_module
from mimir.client.query_engine.history import merge_consecutive_user_messages
from mimir.tests._fake_backend import ScriptedBackend
from mimir.tests.test_agent_loop import RunAgentQueryNonInteractiveTests


def _is_empty_assistant(m: dict) -> bool:
    return (m.get("role") == "assistant"
            and not (m.get("content") or "").strip()
            and not m.get("tool_calls"))


class _LoopRunner(unittest.TestCase):
    """Drives the real agent loop against a scripted backend."""

    def _run(self, script, *, max_steps=8):
        agent = RunAgentQueryNonInteractiveTests._query_agent(self, {"n": 0})
        backend = ScriptedBackend(script)
        emitted: list[dict] = []

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: k["tools"]), \
             patch.object(m, "emit", lambda ev: emitted.append(ev)), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(m.run_agent_query(agent=agent, query="q", max_steps=max_steps))
        # What the session would persist for this turn.
        return result, backend, emitted, list(agent._last_full_messages)


class EmptyTurnDebrisTests(_LoopRunner):
    def test_a_recovered_empty_turn_leaves_nothing_behind(self) -> None:
        result, _, _, history = self._run([{"content": ""}, {"content": "the answer"}])
        self.assertEqual(result, "the answer")
        self.assertEqual([m for m in history if _is_empty_assistant(m)], [])

    def test_an_exhausted_budget_leaves_nothing_behind(self) -> None:
        """The regression: nine empty turns and 21 identical reminders in one session."""
        result, _, _, history = self._run([{"content": ""}] * 10)
        self.assertIn("empty turns", result)
        self.assertEqual([m for m in history if _is_empty_assistant(m)], [])
        self.assertEqual(
            [m for m in history
             if agent_loop_module.EMPTY_TURN_OPENING in str(m.get("content", ""))],
            [],
        )

    def test_the_model_is_not_asked_again_past_the_budget(self) -> None:
        """Exhaustion used to fall through into the nudge and handback paths.

        Each of those injects and `continue`s, so a model that had stopped producing
        anything was run several more times, leaving one more empty turn behind each
        time — the 252/254/256 cluster in the session that prompted this.
        """
        _, backend, _, _ = self._run([{"content": ""}] * 10)
        self.assertEqual(len(backend.calls), AGENT_EMPTY_TURN_RETRIES + 1)

    def test_the_empty_turn_is_diagnosed(self) -> None:
        _, _, emitted, _ = self._run([{"content": ""}, {"content": "the answer"}])
        notes = [e for e in emitted
                 if e.get("type") == "status" and "Empty turn diagnostics" in e.get("text", "")]
        self.assertEqual(len(notes), 1)
        text = notes[0]["text"]
        self.assertIn("messages", text)
        self.assertIn("finish_reason=", text)
        self.assertIn("tail:", text)

    def test_the_diagnostic_names_the_reminders_in_the_prompt(self) -> None:
        """Neither trace tied 'a reminder fired' to 'the turn came back empty'."""
        _, _, emitted, _ = self._run([{"content": ""}, {"content": ""}, {"content": "ok"}])
        notes = [e["text"] for e in emitted
                 if e.get("type") == "status" and "Empty turn diagnostics" in e.get("text", "")]
        self.assertEqual(len(notes), 2)
        # The second empty turn was produced by a prompt that carried the retry reminder.
        self.assertIn("reminders in prompt=1", notes[1])
        self.assertIn("empty_turn", notes[1])


class TransientReminderTests(_LoopRunner):
    def test_a_reminder_reaches_the_call_it_was_injected_for(self) -> None:
        _, backend, _, history = self._run([{"content": ""}, {"content": "the answer"}])
        retry = backend.calls[1]["messages"]
        self.assertIn(agent_loop_module.EMPTY_TURN_OPENING, retry[-1]["content"])
        # …and is gone from what the session keeps.
        self.assertNotIn(agent_loop_module.EMPTY_TURN_OPENING,
                         [m.get("content") for m in history])

    def test_the_event_survives_the_removal(self) -> None:
        """The diagnosis lives in the transcript, not in the message list."""
        emitted: list[dict] = []
        messages: list[dict] = []
        ctx: dict = {}
        with patch("mimir.client.guardrails.nudges.engine.emit", emitted.append):
            inject_reminder(messages, "do the thing", category="empty_turn",
                            tagged=False, execution_context=ctx, step=4)
        drop_transient_reminders(messages, ctx)
        self.assertEqual(messages, [])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["category"], "empty_turn")
        self.assertEqual(emitted[0]["step"], 4)
        self.assertEqual(emitted[0]["text"], "do the thing")

    def test_removal_is_by_identity_not_by_position(self) -> None:
        """The pin sits after the reminders at call time; a tail pop would miss them."""
        messages: list[dict] = [{"role": "user", "content": "q"}]
        ctx: dict = {}
        with patch("mimir.client.guardrails.nudges.engine.emit", lambda ev: None):
            inject_reminder(messages, "first", category="a", tagged=False, execution_context=ctx)
            inject_reminder(messages, "second", category="b", tagged=False, execution_context=ctx)
        messages.append({"role": "user", "content": "the pin"})
        self.assertEqual(drop_transient_reminders(messages, ctx), 2)
        self.assertEqual([m["content"] for m in messages], ["q", "the pin"])

    def test_a_reminder_compaction_already_removed_is_not_an_error(self) -> None:
        messages: list[dict] = []
        ctx: dict = {}
        with patch("mimir.client.guardrails.nudges.engine.emit", lambda ev: None):
            inject_reminder(messages, "gone", category="a", tagged=False, execution_context=ctx)
        messages.clear()  # as intra-query compaction would
        self.assertEqual(drop_transient_reminders(messages, ctx), 0)

    def test_without_a_context_the_reminder_simply_stays(self) -> None:
        messages: list[dict] = []
        with patch("mimir.client.guardrails.nudges.engine.emit", lambda ev: None):
            inject_reminder(messages, "kept", category="a", tagged=False)
        self.assertEqual(len(messages), 1)


class SkillContextTests(unittest.TestCase):
    """The skill block belongs in the system message, not appended after the query."""

    def _messages_for(self, history):
        agent = RunAgentQueryNonInteractiveTests._query_agent(self, {"n": 0})
        agent.skills = {"refactor": {"content": "METHOD."}}
        agent.detect_skill_implicit = None
        seen: dict = {}
        backend = ScriptedBackend([{"content": "done"}])

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: k["tools"]), \
             patch.object(m, "emit", lambda ev: None), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            asyncio.run(m.run_agent_query(agent=agent, query="/refactor it",
                                          history=history, max_steps=4))
        seen["messages"] = backend.calls[0]["messages"]
        return seen["messages"]

    def test_it_lands_in_the_system_message(self) -> None:
        messages = self._messages_for([])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("SKILL CONTEXT", messages[0]["content"])
        self.assertIn("METHOD.", messages[0]["content"])

    def test_it_is_never_a_second_system_message(self) -> None:
        """Nine copies accumulated in one session because it was appended, not folded."""
        messages = self._messages_for([])
        self.assertEqual([m for m in messages[1:] if m.get("role") == "system"], [])

    def test_a_second_query_does_not_stack_another_copy(self) -> None:
        first = self._messages_for([])
        carried = [m for m in first[1:]]  # what a session hands back, system excluded
        second = self._messages_for(carried)
        self.assertEqual([m for m in second[1:] if m.get("role") == "system"], [])
        self.assertEqual(second[0]["content"].count("SKILL CONTEXT"), 1)


class SharedRoleNormalizationTests(unittest.TestCase):
    """Every backend gets the merge; it used to live inside the vLLM one."""

    def test_adjacent_user_turns_are_merged(self) -> None:
        out = merge_consecutive_user_messages([
            {"role": "user", "content": "q"},
            {"role": "user", "content": "reminder"},
        ])
        self.assertEqual(out, [{"role": "user", "content": "q\n\nreminder"}])

    def test_an_identical_duplicate_is_dropped(self) -> None:
        out = merge_consecutive_user_messages([
            {"role": "user", "content": "same"},
            {"role": "user", "content": "same"},
        ])
        self.assertEqual(out, [{"role": "user", "content": "same"}])

    def test_the_anthropic_path_applies_it(self) -> None:
        from mimir.client.query_engine.backends.anthropic_backend import AnthropicBackend

        _system, out = AnthropicBackend()._prepare([
            {"role": "user", "content": "q"},
            {"role": "user", "content": "reminder"},
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["content"][0]["text"], "q\n\nreminder")

    def test_the_ollama_path_applies_it(self) -> None:
        import mimir.client.query_engine.backends.ollama_backend as ob

        sent: dict = {}

        def _fake_chat(**kw):
            sent["messages"] = kw["messages"]
            return {"message": {"role": "assistant", "content": "hi"}}

        with patch.object(ob.ollama, "chat", _fake_chat):
            ob.OllamaBackend().chat(
                "m",
                [{"role": "user", "content": "q"}, {"role": "user", "content": "reminder"}],
                [], False, False, {"num_ctx": 4096}, token_callback=lambda t: None,
            )
        self.assertEqual(sent["messages"], [{"role": "user", "content": "q\n\nreminder"}])


if __name__ == "__main__":
    unittest.main()
