"""A connection that drops under a running turn must not cost the turn.

One ``_AgentWorker`` serves the whole server and outlives every connection, but a
``_Session`` is per-socket. A drop mid-run therefore left a turn alive with nobody
reading it, and the next connection did three things to it: it emptied the queue the
turn was writing into, it never put back the card the turn was parked on, and it told
the client nothing was running. The turn then waited on an answer that could not
arrive — the query loop is serial, so every later query queued behind that wait — and
the session read as hung with nothing on screen to say why.
"""
import json
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests.test_session_isolation import _FakeWS, _bare_worker, SessionFencingTests


def _parked_worker(prompt: dict | None = None) -> _AgentWorker:
    w = _bare_worker()
    w._pending_prompt = prompt
    return w


class PendingPromptTests(unittest.TestCase):
    def test_emitting_a_card_marks_the_turn_parked_on_it(self):
        w = _parked_worker()
        card = {"type": "user_question", "id": "q1", "questions": []}
        w._emit_prompt(card)
        self.assertEqual(w.out_q.get_nowait(), card)
        self.assertEqual(w.pending_prompt(), card)

    def test_the_card_is_a_copy_of_what_was_sent(self):
        """The queued dict is drained and stamped downstream; the record must not follow."""
        w = _parked_worker()
        card = {"type": "approval", "id": "a1"}
        w._emit_prompt(card)
        w.out_q.get_nowait()["session_id"] = "s1"
        self.assertEqual(w.pending_prompt(), {"type": "approval", "id": "a1"})

    def test_an_answer_ends_the_park(self):
        w = _parked_worker({"type": "approval", "id": "a1"})
        w._agent = None
        w._approval_q.put({"choice": "y"})
        self.assertEqual(w._await_response(w._approval_q), {"choice": "y"})
        self.assertIsNone(w.pending_prompt())

    def test_a_cancelled_wait_ends_the_park(self):
        class _Cancelled:
            class _Flag:
                @staticmethod
                def is_set():
                    return True
            _cancel_flag = _Flag()

        w = _parked_worker({"type": "approval", "id": "a1"})
        w._agent = _Cancelled()
        self.assertIsNone(w._await_response(w._approval_q))
        self.assertIsNone(w.pending_prompt())

    def test_a_card_still_on_its_way_out_is_not_handed_out_twice(self):
        """The drain loop will deliver it; resending it too is a card answered twice."""
        w = _parked_worker()
        w._emit_prompt({"type": "approval", "id": "a1"})
        self.assertIsNone(w.pending_prompt())
        w.out_q.get_nowait()                      # delivered to whoever was connected
        self.assertEqual(w.pending_prompt(), {"type": "approval", "id": "a1"})

    def test_abandoning_the_turn_forgets_the_card(self):
        w = _parked_worker({"type": "continue_prompt", "id": "c1", "summary": "3 left"})
        w.flush_prompts()
        self.assertIsNone(w.pending_prompt())


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, worker):
        return SessionFencingTests._session(self, worker)

    async def test_the_card_a_turn_is_parked_on_comes_back(self):
        card = {"type": "user_question", "id": "q1", "questions": [{"header": "Plan approval"}]}
        sess = self._session(_parked_worker(card))
        await sess._resend_parked_prompt()
        self.assertEqual([json.loads(p) for p in sess.ws.sent], [card])

    async def test_nothing_is_sent_when_no_turn_is_parked(self):
        sess = self._session(_parked_worker(None))
        await sess._resend_parked_prompt()
        self.assertEqual(sess.ws.sent, [])

    async def test_a_socket_that_fails_on_the_resend_does_not_break_the_connection(self):
        sess = self._session(_parked_worker({"type": "approval", "id": "a1"}))

        async def _boom(_payload):
            raise ConnectionResetError("gone")

        sess.ws.send = _boom
        await sess._resend_parked_prompt()  # must not raise

    def test_a_busy_worker_keeps_the_events_of_its_running_turn(self):
        w = _parked_worker()
        w.is_busy = lambda: True
        w.out_q.put({"type": "tool_call", "id": "t1"})
        w.out_q.put({"type": "answer", "text": "what I did"})
        self._session(w)._drop_stale_events()
        self.assertEqual(w.out_q.qsize(), 2)

    def test_an_idle_worker_leaves_only_debris_behind(self):
        w = _parked_worker()
        w.is_busy = lambda: False
        w.out_q.put({"type": "status", "text": "left over"})
        self._session(w)._drop_stale_events()
        self.assertTrue(w.out_q.empty())


if __name__ == "__main__":
    unittest.main()


class AgentReadinessTests(unittest.TestCase):
    """The socket opens long before the agent exists, and both greetings say "ready".

    ``_Session.run`` sends one the moment the socket is accepted, so the webview can
    leave "connecting"; the worker sends its own only after the LLM backend answers
    and the agent is constructed, which on a cold vLLM is minutes later. Nothing in
    either payload told them apart, so the chat showed "Type a message to start" over
    an agent that could not yet answer one. The flag is what the greeting states.
    """

    def test_a_worker_without_an_agent_is_not_ready(self):
        w = _bare_worker()
        w._agent = None
        self.assertFalse(w.agent_ready())

    def test_a_worker_with_an_agent_is_ready(self):
        w = _bare_worker()
        w._agent = object()
        self.assertTrue(w.agent_ready())


class GreetingTests(unittest.IsolatedAsyncioTestCase):
    """The greeting the socket sends, built by the code that really sends it."""

    def _session(self, agent):
        w = _bare_worker()
        w._agent = agent
        w.model = "m"
        w.get_context_mode = lambda: "compact"
        w.get_enforcement = lambda: "strict"
        w.get_approval_mode = lambda: "normal"
        w.get_thinking_profile = lambda: {}
        w.get_temperature_state = lambda: {"supported": True, "value": None}
        return SessionFencingTests._session(self, w)

    def test_the_socket_greeting_admits_the_agent_is_not_up(self):
        self.assertFalse(self._session(None)._greeting()["agent_ready"])

    def test_the_socket_greeting_reports_a_live_agent(self):
        self.assertTrue(self._session(object())._greeting()["agent_ready"])

    async def test_run_sends_that_greeting_verbatim(self):
        """Guards the seam: a greeting built correctly and sent as something else."""
        sess = self._session(None)

        async def _stop(_payload):
            sess.ws.sent.append(_payload)
            raise ConnectionResetError("caller went away")

        sess.ws.send = _stop
        await sess.run()
        self.assertFalse(json.loads(sess.ws.sent[0])["agent_ready"])


class _NoSessions:
    """A store with nothing archived, so ``run`` takes the new-session path."""

    @staticmethod
    def list_sessions():
        return []


class _ClosedWS(_FakeWS):
    """A socket that accepts sends and carries no inbound messages."""

    def __aiter__(self):
        async def _empty():
            return
            yield  # pragma: no cover - makes this an async generator

        return _empty()


class ReadinessAnnouncementTests(unittest.IsolatedAsyncioTestCase):
    """The agent came up while the socket was being set up, and nobody was told.

    The worker announces its readiness exactly once, by queueing a second ``ready``
    on ``out_q``. ``run`` empties that queue right after greeting the client, on the
    grounds that an idle worker's queue is debris — and an agent that has just
    finished starting is idle. The announcement fell in that gap, the client kept the
    ``agent_ready: False`` it was greeted with, and the chat offered "starting the
    agent — waiting for the model backend" over an agent that was already up. It
    stayed there as long as the socket lived: nothing repeats the announcement.
    """

    def _session(self, worker):
        sess = SessionFencingTests._session(self, worker)
        sess.ws = _ClosedWS()
        # The session init between the greeting and the listen loop is not what is
        # under test; what matters is that the readiness question is asked after it.
        sess._purge_empty_sessions = lambda: None

        async def _noop(*_a, **_kw):
            return None

        sess._send_sessions_list = _noop
        sess._send_toggles = _noop
        sess._create_new_session = _noop
        sess._resend_parked_prompt = _noop
        sess._drain_loop = _noop
        sess._summary_task = None
        sess.store = _NoSessions()
        return sess

    def _worker(self, agent):
        w = _bare_worker()
        w._agent = agent
        w.model = "m"
        w.is_busy = lambda: False
        w.get_context_mode = lambda: "full"
        w.get_enforcement = lambda: "light"
        w.get_approval_mode = lambda: "manual"
        w.get_thinking_profile = lambda: {}
        w.get_temperature_state = lambda: {"supported": True, "value": None}
        return w

    async def test_an_agent_that_comes_up_during_setup_is_announced(self):
        w = self._worker(None)
        sess = self._session(w)
        # The gap itself: the worker finishes starting, queues its one announcement,
        # and the stale-event purge throws it away.
        original = sess._drop_stale_events

        def _ready_then_purge():
            w._agent = object()
            w.out_q.put({"type": "ready", "model": "m", "agent_ready": True})
            original()

        sess._drop_stale_events = _ready_then_purge

        await sess.run()

        greetings = [json.loads(p) for p in sess.ws.sent if json.loads(p)["type"] == "ready"]
        self.assertFalse(greetings[0]["agent_ready"])   # true when the socket opened
        self.assertTrue(greetings[-1]["agent_ready"])   # true by the time it listens

    async def test_an_agent_still_starting_is_not_announced_as_ready(self):
        sess = self._session(self._worker(None))
        await sess.run()
        greetings = [json.loads(p) for p in sess.ws.sent if json.loads(p)["type"] == "ready"]
        self.assertEqual(len(greetings), 1)
        self.assertFalse(greetings[0]["agent_ready"])

    async def test_an_agent_up_before_the_socket_is_greeted_once(self):
        sess = self._session(self._worker(object()))
        await sess.run()
        greetings = [json.loads(p) for p in sess.ws.sent if json.loads(p)["type"] == "ready"]
        self.assertEqual(len(greetings), 1)
        self.assertTrue(greetings[0]["agent_ready"])
