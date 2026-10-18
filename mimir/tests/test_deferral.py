"""A turn waiting on the user survives the user leaving its conversation.

Leaving disturbs nothing: the turn keeps running, and its card stays up carrying the
conversation that raised it, so a plan up for approval or a command waiting for its
go-ahead is still there when the user comes back.

Deferral is for the cases where the wait really must be set aside — the user moving past a
card, a conversation deleted while one of its turns is parked. The card is stored with the
conversation and answering it resumes the turn from the exact call it stopped on. Both
halves are covered here: the worker's side of setting a wait aside, and the session's side
of resuming on the answer.
"""
from __future__ import annotations

import asyncio
import threading
import types
import unittest

from mimir.client.event_sink import event_sink as capture_events
from mimir.tests.support.pools import FakePool
from mimir.tests.support.connections import bare_session
from unittest.mock import patch

from mimir.client.query_engine import agent_loop as agent_loop_module
from mimir.client.query_engine import streaming as streaming_module
from mimir.client.query_engine.deferral import (
    CURRENT_CALL_ID,
    DEFERRED_RESULT,
    KIND_CALLS,
    KIND_PLAN_DECISION,
    end_deferred_calls,
    take_deferred_calls,
)
from mimir.client.ui.ws.connection.connection import Connection
from mimir.client.ui.ws.worker.worker import AgentWorker
from mimir.tests.support.backend import ScriptedBackend
from mimir.tests.test_agent_loop import RunAgentQueryNonInteractiveTests, _tool_call
import mimir.client.query_engine.loop_base as loop_base_module
import mimir.client.query_engine.turn as turn_module
from mimir.tests.support.loops import run_plan
from mimir.client.core.turn_state import TurnState
from mimir.tests.support.agents import fake_agent

_PROMPT = {"type": "approval", "id": "card-1", "tool": "bash_run"}


def _worker() -> AgentWorker:
    w = AgentWorker("test-model", start=False)
    w._current_task = object()
    w.session_id = "s1"
    w.query_session_id = "s1"
    w._agent = types.SimpleNamespace(turn=TurnState(cancel_flag=threading.Event(), deferred_prompts=[]))
    return w


class WorkerDeferralTests(unittest.TestCase):
    def test_a_deferred_wait_returns_and_names_its_call(self) -> None:
        w = _worker()
        w.human.emit_prompt(dict(_PROMPT))
        self.assertTrue(w.human.defer_turn())
        token = CURRENT_CALL_ID.set("call-7")
        try:
            self.assertIsNone(w.human.await_response(_PROMPT["id"]))
        finally:
            CURRENT_CALL_ID.reset(token)
        self.assertEqual(
            w.agent.turn.deferred_prompts,
            [{"call_id": "call-7",
              "prompt": {**_PROMPT, "session_id": "s1", "session_title": ""},
              "questions": []}])

    def test_nothing_to_set_aside_when_not_parked(self) -> None:
        self.assertFalse(_worker().human.defer_turn())

    def test_a_card_raised_while_deferring_is_not_sent(self) -> None:
        w = _worker()
        w.human._defer.set()
        w.human.emit_prompt(dict(_PROMPT))
        self.assertTrue(w.out_q.empty())

    def test_a_resume_answers_its_card_without_showing_it(self) -> None:
        w = _worker()
        w.human._preanswer = {"type": "approval", "response": {"choice": "y"}}
        w.human.emit_prompt(dict(_PROMPT))
        self.assertTrue(w.out_q.empty())
        self.assertEqual(w.human.await_response(_PROMPT["id"]), {"choice": "y"})
        # One answer, one card: the next one is asked for real.
        self.assertIsNone(w.human._preanswer)
        w.human.emit_prompt(dict(_PROMPT))
        self.assertFalse(w.out_q.empty())


class LoopDeferralTests(unittest.TestCase):
    """The step ends on the deferred call, and its answer picks the step back up."""

    def _run(self, agent, script, dispatch, **kwargs):
        backend = ScriptedBackend(script)

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(loop_base_module, "dispatch_tool_calls", dispatch), \
             patch.object(m, "tools_for_context", lambda **k: k["tools"]), \
             capture_events(lambda ev: None), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(turn_module.run_agent_query(agent=agent, query="q", **kwargs))
        return result, backend

    def test_a_deferred_call_ends_the_turn_and_resumes_on_its_answer(self) -> None:
        agent = RunAgentQueryNonInteractiveTests._query_agent(self)
        agent.turn.deferred_prompts = []

        async def _defer(tool_calls, agent, messages, execution_context):
            for tc in tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "denied"})
            agent.turn.deferred_prompts = [{"call_id": "a", "prompt": _PROMPT}]

        result, backend = self._run(
            agent, [{"content": "", "tool_calls": [_tool_call("bash_run", call_id="a")]}],
            _defer, history=[{"role": "user", "content": "q"}])
        self.assertEqual(result, "")
        self.assertEqual(len(backend.calls), 1)  # no model call after the deferral
        record = agent.turn.deferred_turn
        self.assertEqual((record["kind"], record["call_ids"]), (KIND_CALLS, ["a"]))
        parked = [m for m in agent.turn.last_full_messages if m.get("role") == "tool"]
        self.assertEqual(parked[0]["content"], DEFERRED_RESULT)

        # The user comes back and approves: the call runs, then the model continues.
        history = list(agent.turn.last_full_messages)
        agent.turn.deferred_prompts = []
        agent.turn.deferred_turn = None
        ran = []

        async def _run_it(tool_calls, agent, messages, execution_context):
            for tc in tool_calls:
                ran.append(tc["id"])
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "done"})

        result, backend = self._run(agent, [{"content": "all good"}], _run_it,
                                    history=history, resume=record)
        self.assertEqual(result, "all good")
        self.assertEqual(ran, ["a"])
        sent = backend.calls[0]["messages"]
        self.assertEqual([m["content"] for m in sent if m["role"] == "tool"], ["done"])
        # No new user message: the resumed turn is the same turn.
        self.assertEqual([m["content"] for m in sent if m["role"] == "user"], ["q"])


class PlanDecisionDeferralTests(unittest.TestCase):
    def _run_plan(self, ask):
        from mimir.client.context.execution_context import build_execution_context
        from mimir.client.query_engine import plan_loop as plan_loop_module

        backend = ScriptedBackend([])
        agent = fake_agent(model="m", tools=[], tool_caps={}, request_user_question=None, turn=TurnState(deferred_prompts=[]))
        agent.request_user_question = lambda questions: ask(agent)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"},
                    {"role": "assistant", "content": "Here is the plan."}]
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps, **kw: []), \
             capture_events(lambda ev: None), \
             patch.object(loop_base_module, "finalize_answer", _finalize):
            result = asyncio.run(run_plan(
                agent=agent, query="plan it", messages=messages,
                execution_context=build_execution_context(),
                max_steps=10, thinking=False, streaming=False, logger=None,
                cb={"think_token_callback": None}, resume_decision=True,
            ))
        return result, backend, agent, messages

    def test_the_decision_is_asked_again_without_a_model_call(self) -> None:
        from mimir.client.query_engine.plan_loop import _PLAN_REJECT
        from mimir.client.guardrails.nudges.messages import PLAN_REJECTED_ANSWER

        result, backend, _, _ = self._run_plan(
            lambda agent: {"answers": [{"selected": [_PLAN_REJECT]}]})
        self.assertEqual(result, PLAN_REJECTED_ANSWER)
        self.assertEqual(backend.calls, [])

    def test_leaving_again_defers_the_decision_again(self) -> None:
        def _leave(agent):
            agent.turn.deferred_prompts = [{"call_id": None, "prompt": {"id": "q1"}}]
            return {"answers": []}

        _, backend, agent, _ = self._run_plan(_leave)
        self.assertEqual(agent.turn.deferred_turn["kind"], KIND_PLAN_DECISION)
        self.assertEqual(backend.calls, [])


class AttributionTests(unittest.TestCase):
    def test_a_question_is_matched_to_the_call_that_carries_it(self) -> None:
        agent = fake_agent(turn=TurnState(deferred_prompts=[
            {"call_id": None, "prompt": {}, "questions": [{"question": "Quelle base ?"}]}]))
        calls = [_tool_call("read_file", '{"path": "x"}', call_id="r"),
                 _tool_call("ask", '{"questions": [{"question": "Quelle base ?"}]}', call_id="q")]
        messages = [{"role": "tool", "tool_call_id": "r", "content": "x"},
                    {"role": "tool", "tool_call_id": "q", "content": "declined"}]
        self.assertTrue(end_deferred_calls(agent, messages, calls, query="q", mode="agent"))
        self.assertEqual(agent.turn.deferred_turn["call_ids"], ["q"])
        self.assertEqual(messages[0]["content"], "x")
        self.assertEqual(messages[1]["content"], DEFERRED_RESULT)

    def test_taking_the_calls_removes_only_their_placeholders(self) -> None:
        messages = [{"role": "assistant", "content": "",
                     "tool_calls": [{"id": "r"}, {"id": "q"}]},
                    {"role": "tool", "tool_call_id": "r", "content": "x"},
                    {"role": "tool", "tool_call_id": "q", "content": DEFERRED_RESULT}]
        self.assertEqual(take_deferred_calls(messages, ["q"]), [{"id": "q"}])
        self.assertEqual([m.get("tool_call_id") for m in messages], [None, "r"])


class _FakeWorker:
    def __init__(self, parked: bool) -> None:
        self.human = self   # plays its own bridge: the answers are what is recorded
        self.parked = parked
        self.calls: list = []
        self.query_session_id = "s1"
        self.resumed = None

    def is_busy(self) -> bool:
        return True

    def defer(self) -> bool:
        self.calls.append("defer")
        return self.parked

    def cancel(self) -> bool:
        self.calls.append("cancel")
        self.parked = None
        return True

    def flush_prompts(self) -> None:
        self.calls.append("flush")

    def submit_resume(self, record, answer, history, session_id=None) -> None:
        self.resumed = (record, answer, history, session_id)

    def resolve_approval(self, choice, approved_files=None, req_id="") -> None:
        # The id travels now: an answer is delivered to the card it names, not to
        # whatever happens to be listening.
        self.calls.append(("resolve", choice, req_id))


def _session(worker) -> Connection:
    sess = bare_session(FakePool(worker))
    sess.screen.save = lambda: None
    return sess


async def _noop(*_a, **_k):
    return None


class SessionDeferralTests(unittest.IsolatedAsyncioTestCase):
    async def test_leaving_a_parked_turn_leaves_it_parked(self) -> None:
        """The card carries the conversation that raised it, so it stays up and is
        answered there — nothing has to be set aside to keep it reachable."""
        w = _FakeWorker(parked=True)
        sess = _session(w)
        sess.pool.conversations.new("s2")
        sess.screen.send_sessions_list = _noop
        sess.conversations.load_session = _noop
        await sess.conversations.on_switch_session({"session_id": "s2"})
        self.assertEqual(w.calls, [])

    async def test_leaving_a_working_turn_leaves_it_working(self) -> None:
        w = _FakeWorker(parked=False)
        sess = _session(w)
        sess.pool.conversations.new("s2")
        sess.screen.send_sessions_list = _noop
        await sess.conversations.on_create_session({})
        self.assertEqual(w.calls, [])

    async def test_answering_the_card_resumes_the_turn(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        record = {"kind": KIND_CALLS, "call_ids": ["a"], "prompt": dict(_PROMPT)}
        sess.view.conv.pending_interaction = record
        placeholder = {"role": "tool", "tool_call_id": "a", "content": DEFERRED_RESULT}
        sess.view.conv.history = [{"role": "assistant", "tool_calls": [{"id": "a"}]},
                             dict(placeholder)]
        sess.view.conv.record.llm_history_full = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]}, dict(placeholder)]
        await sess.cards.on_approval_response({"id": "card-1", "choice": "y"})
        rec, answer, history, sid = w.resumed
        self.assertEqual((rec, sid), (record, "s1"))
        self.assertEqual(answer["choice"], "y")
        self.assertEqual(len(history), 1)          # placeholder gone from the window
        self.assertEqual(len(sess.view.conv.history_full), 1)  # …and from the record
        self.assertIsNone(sess.view.conv.pending_interaction)
        self.assertFalse([c for c in w.calls if c[0] == "resolve"])

    async def test_a_live_card_is_still_answered_live(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        await sess.cards.on_approval_response(
            {"id": "other", "choice": "n", "session_id": "s1"})
        # Answered live, and the card it names travels with it.
        self.assertEqual(w.calls, [("resolve", "n", "other")])

    async def test_an_answer_naming_no_conversation_is_dropped(self) -> None:
        """The worst failure this layer can produce is settling a question a different
        conversation asked, with the user's approval attached to a call they never saw. So
        an unattributed answer goes nowhere — never to whoever is on screen."""
        w = _FakeWorker(parked=True)
        sess = _session(w)
        await sess.cards.on_approval_response({"id": "other", "choice": "y"})
        self.assertEqual(w.calls, [])

    async def test_an_answer_for_a_conversation_with_no_agent_is_dropped(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        sess.pool.worker = None
        await sess.cards.on_approval_response(
            {"id": "other", "choice": "y", "session_id": "gone"})
        self.assertEqual(w.calls, [])

    async def test_an_answer_to_a_card_moved_past_goes_nowhere(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        sess.view.stale_prompt_ids = {"card-1"}
        await sess.cards.on_approval_response({"id": "card-1", "choice": "y"})
        self.assertEqual(w.calls, [])
        self.assertIsNone(w.resumed)


if __name__ == "__main__":
    unittest.main()
