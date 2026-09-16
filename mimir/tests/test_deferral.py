"""A turn waiting on the user survives the user leaving its conversation.

Switching session used to cancel the turn outright: a plan up for approval, or a
command waiting for its go-ahead, was gone when the user came back. Now the wait is
set aside (``query_engine.deferral``), the card is stored with the conversation, and
answering it resumes the turn from the exact call it stopped on.
"""
from __future__ import annotations

import asyncio
import queue as _queue
import threading
import types
import unittest
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
from mimir.client.ui.ws.ws_session import _Session
from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests._fake_backend import ScriptedBackend
from mimir.tests.test_agent_loop import RunAgentQueryNonInteractiveTests, _tool_call

_PROMPT = {"type": "approval", "id": "card-1", "tool": "bash_run"}


def _worker() -> _AgentWorker:
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w._approval_q = _queue.Queue()
    w._question_q = _queue.Queue()
    w._pending_prompt = None
    w._pending_questions = None
    w._defer = threading.Event()
    w._preanswer = None
    w._current_task = object()
    w._agent = types.SimpleNamespace(_cancel_flag=threading.Event(), _deferred_prompts=[])
    return w


class WorkerDeferralTests(unittest.TestCase):
    def test_a_deferred_wait_returns_and_names_its_call(self) -> None:
        w = _worker()
        w._emit_prompt(dict(_PROMPT))
        self.assertTrue(w.defer())
        token = CURRENT_CALL_ID.set("call-7")
        try:
            self.assertIsNone(w._await_response(w._approval_q))
        finally:
            CURRENT_CALL_ID.reset(token)
        self.assertEqual(w._agent._deferred_prompts,
                         [{"call_id": "call-7", "prompt": _PROMPT, "questions": []}])

    def test_nothing_to_set_aside_when_not_parked(self) -> None:
        self.assertFalse(_worker().defer())

    def test_a_card_raised_while_deferring_is_not_sent(self) -> None:
        w = _worker()
        w._defer.set()
        w._emit_prompt(dict(_PROMPT))
        self.assertTrue(w.out_q.empty())

    def test_a_resume_answers_its_card_without_showing_it(self) -> None:
        w = _worker()
        w._preanswer = {"type": "approval", "response": {"choice": "y"}}
        w._emit_prompt(dict(_PROMPT))
        self.assertTrue(w.out_q.empty())
        self.assertEqual(w._await_response(w._approval_q), {"choice": "y"})
        # One answer, one card: the next one is asked for real.
        self.assertIsNone(w._preanswer)
        w._emit_prompt(dict(_PROMPT))
        self.assertFalse(w.out_q.empty())


class LoopDeferralTests(unittest.TestCase):
    """The step ends on the deferred call, and its answer picks the step back up."""

    def _run(self, agent, script, dispatch, **kwargs):
        backend = ScriptedBackend(script)

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(m, "_dispatch_tool_calls", dispatch), \
             patch.object(m, "tools_for_context", lambda **k: k["tools"]), \
             patch.object(m, "emit", lambda ev: None), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(m.run_agent_query(agent=agent, query="q", **kwargs))
        return result, backend

    def test_a_deferred_call_ends_the_turn_and_resumes_on_its_answer(self) -> None:
        agent = RunAgentQueryNonInteractiveTests._query_agent(self)
        agent._deferred_prompts = []

        async def _defer(tool_calls, agent, messages, execution_context):
            for tc in tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "denied"})
            agent._deferred_prompts = [{"call_id": "a", "prompt": _PROMPT}]

        result, backend = self._run(
            agent, [{"content": "", "tool_calls": [_tool_call("bash_run", call_id="a")]}],
            _defer, history=[{"role": "user", "content": "q"}])
        self.assertEqual(result, "")
        self.assertEqual(len(backend.calls), 1)  # no model call after the deferral
        record = agent._deferred_turn
        self.assertEqual((record["kind"], record["call_ids"]), (KIND_CALLS, ["a"]))
        parked = [m for m in agent._last_full_messages if m.get("role") == "tool"]
        self.assertEqual(parked[0]["content"], DEFERRED_RESULT)

        # The user comes back and approves: the call runs, then the model continues.
        history = list(agent._last_full_messages)
        agent._deferred_prompts = []
        agent._deferred_turn = None
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
        agent = types.SimpleNamespace(model="m", tools=[], tool_caps={},
                                      _deferred_prompts=[], _request_user_question=None)
        agent._request_user_question = lambda questions: ask(agent)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"},
                    {"role": "assistant", "content": "Here is the plan."}]
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps, **kw: []), \
             patch.object(plan_loop_module, "emit", lambda ev: None), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            result = asyncio.run(plan_loop_module._run_plan_mode(
                agent=agent, query="plan it", messages=messages,
                execution_context=build_execution_context(),
                max_steps=10, thinking=False, streaming=False, logger=None,
                cb={"think_token_callback": None}, resume_decision=True,
            ))
        return result, backend, agent, messages

    def test_the_decision_is_asked_again_without_a_model_call(self) -> None:
        from mimir.client.query_engine.plan_loop import _PLAN_REJECT
        from mimir.client.guardrails.workflow import PLAN_REJECTED_ANSWER

        result, backend, _, _ = self._run_plan(
            lambda agent: {"answers": [{"selected": [_PLAN_REJECT]}]})
        self.assertEqual(result, PLAN_REJECTED_ANSWER)
        self.assertEqual(backend.calls, [])

    def test_leaving_again_defers_the_decision_again(self) -> None:
        def _leave(agent):
            agent._deferred_prompts = [{"call_id": None, "prompt": {"id": "q1"}}]
            return {"answers": []}

        _, backend, agent, _ = self._run_plan(_leave)
        self.assertEqual(agent._deferred_turn["kind"], KIND_PLAN_DECISION)
        self.assertEqual(backend.calls, [])


class AttributionTests(unittest.TestCase):
    def test_a_question_is_matched_to_the_call_that_carries_it(self) -> None:
        agent = types.SimpleNamespace(_deferred_prompts=[
            {"call_id": None, "prompt": {}, "questions": [{"question": "Quelle base ?"}]}])
        calls = [_tool_call("read_file", '{"path": "x"}', call_id="r"),
                 _tool_call("ask", '{"questions": [{"question": "Quelle base ?"}]}', call_id="q")]
        messages = [{"role": "tool", "tool_call_id": "r", "content": "x"},
                    {"role": "tool", "tool_call_id": "q", "content": "declined"}]
        self.assertTrue(end_deferred_calls(agent, messages, calls, query="q", mode="agent"))
        self.assertEqual(agent._deferred_turn["call_ids"], ["q"])
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
        self.parked = parked
        self.calls: list = []
        self._query_session_id = "s1"
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

    def resolve_approval(self, choice, approved_files=None) -> None:
        self.calls.append(("resolve", choice))


def _session(worker) -> _Session:
    sess = object.__new__(_Session)
    sess.worker = worker
    sess._active_session_id = "s1"
    sess._detached_turns = {}
    sess._submitted_len = 3
    sess._pending_interaction = None
    sess._stale_prompt_ids = set()
    sess._display_messages = []
    sess._autosave_session = lambda msgs: None
    return sess


class SessionDeferralTests(unittest.IsolatedAsyncioTestCase):
    async def test_leaving_a_parked_turn_defers_it(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        self.assertEqual(await sess._abandon_running_turn(), "deferred")
        self.assertEqual(w.calls, ["defer"])
        # Its answer is written to the session it belongs to.
        self.assertEqual(sess._detached_turns, {"s1": 3})

    async def test_leaving_a_working_turn_still_cancels_it(self) -> None:
        w = _FakeWorker(parked=False)

        def _idle():
            return w.parked is not None
        w.is_busy = _idle
        sess = _session(w)
        self.assertEqual(await sess._abandon_running_turn(), "cancelled")
        self.assertEqual(w.calls, ["defer", "cancel", "flush"])

    async def test_answering_the_card_resumes_the_turn(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        record = {"kind": KIND_CALLS, "call_ids": ["a"], "prompt": dict(_PROMPT)}
        sess._pending_interaction = record
        placeholder = {"role": "tool", "tool_call_id": "a", "content": DEFERRED_RESULT}
        sess.history = [{"role": "assistant", "tool_calls": [{"id": "a"}]}, dict(placeholder)]
        sess.history_full = [{"role": "assistant", "tool_calls": [{"id": "a"}]}, dict(placeholder)]
        await sess._handle_approval_response({"id": "card-1", "choice": "y"})
        rec, answer, history, sid = w.resumed
        self.assertEqual((rec, sid), (record, "s1"))
        self.assertEqual(answer["choice"], "y")
        self.assertEqual(len(history), 1)          # placeholder gone from the window
        self.assertEqual(len(sess.history_full), 1)  # …and from the record
        self.assertIsNone(sess._pending_interaction)
        self.assertNotIn(("resolve", "y"), w.calls)

    async def test_a_live_card_is_still_answered_live(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        await sess._handle_approval_response({"id": "other", "choice": "n"})
        self.assertEqual(w.calls, [("resolve", "n")])

    async def test_an_answer_to_a_card_moved_past_goes_nowhere(self) -> None:
        w = _FakeWorker(parked=True)
        sess = _session(w)
        sess._stale_prompt_ids = {"card-1"}
        await sess._handle_approval_response({"id": "card-1", "choice": "y"})
        self.assertEqual(w.calls, [])
        self.assertIsNone(w.resumed)


if __name__ == "__main__":
    unittest.main()
