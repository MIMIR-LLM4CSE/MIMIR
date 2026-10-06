"""A connection that drops under a running turn must not cost the turn.

One ``_AgentWorker`` serves the whole server and outlives every connection, but a
``_Session`` is per-socket. A drop mid-run therefore left a turn alive with nobody
reading it, and the next connection did three things to it: it emptied the queue the
turn was writing into, it never put back the card the turn was parked on, and it told
the client nothing was running. The turn then waited on an answer that could not
arrive — the query loop is serial, so every later query queued behind that wait — and
the session read as hung with nothing on screen to say why.
"""
import asyncio
import json
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws import transcript_log
from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests.test_session_isolation import _FakeWS, _bare_worker, SessionFencingTests


def _parked_worker(prompt: dict | None = None, session_id: str = "s1",
                   title: str = "Install the toolchain") -> _AgentWorker:
    w = _bare_worker(session_id=session_id)
    w.session_title = title
    w._pending_prompt = prompt
    return w


def _attributed(card: dict, session_id: str = "s1",
                title: str = "Install the toolchain") -> dict:
    """*card* as _emit_prompt stamps it: carrying the conversation that is asking."""
    return {**card, "session_id": session_id, "session_title": title}


class PendingPromptTests(unittest.TestCase):
    def test_emitting_a_card_marks_the_turn_parked_on_it(self):
        w = _parked_worker()
        card = {"type": "user_question", "id": "q1", "questions": []}
        w._emit_prompt(card)
        self.assertEqual(w.out_q.get_nowait(), _attributed(card))
        self.assertEqual(w.pending_prompt(), _attributed(card))

    def test_a_card_says_which_conversation_is_asking(self):
        """Several conversations can be parked at once, and one may be asking while the
        user reads another. The answer is routed by this, so a card without it is one
        that cannot be answered — and the attribution is structured rather than written
        into the question text, which the model also sees."""
        w = _parked_worker(session_id="s2", title="Port the solver")
        w._emit_prompt({"type": "approval", "id": "a1"})
        card = w.out_q.get_nowait()
        self.assertEqual(card["session_id"], "s2")
        self.assertEqual(card["session_title"], "Port the solver")

    def test_a_turn_resuming_another_conversation_names_that_one(self):
        """A background-job wake runs a turn for the session that launched the job."""
        w = _parked_worker(session_id="s1")
        w._query_session_id = "s2"
        w._emit_prompt({"type": "approval", "id": "a1"})
        self.assertEqual(w.out_q.get_nowait()["session_id"], "s2")

    def test_the_card_is_a_copy_of_what_was_sent(self):
        """The queued dict is drained and stamped downstream; the record must not follow."""
        w = _parked_worker()
        card = {"type": "approval", "id": "a1"}
        w._emit_prompt(card)
        w.out_q.get_nowait()["id"] = "tampered"
        self.assertEqual(w.pending_prompt(), _attributed({"type": "approval", "id": "a1"}))

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
        self.assertEqual(w.pending_prompt(), _attributed({"type": "approval", "id": "a1"}))

    def test_abandoning_the_turn_forgets_the_card(self):
        w = _parked_worker({"type": "continue_prompt", "id": "c1", "summary": "3 left"})
        w.flush_prompts()
        self.assertIsNone(w.pending_prompt())


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # The pump journals what it drains, so these tests write a transcript: give
        # them their own state dir rather than the real one.
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(transcript_log, "_MIMIR_DIR_WS", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

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

    def test_a_reconnect_throws_nothing_away(self):
        # Nothing is swept on connect. The pump drains every worker whether or not a
        # socket exists and the journal holds what it drained, so a reconnect replays
        # from its watermark rather than hoping the right things were kept — and a
        # sweep that had to carve out a finished job's wake was deciding a question
        # that does not need deciding.
        for busy in (True, False):
            with self.subTest(busy=busy):
                sid = f"busy-{busy}"   # its own journal, so the runs do not add up
                w = _parked_worker()
                w.session_id = sid
                w.is_busy = lambda: busy
                w.out_q.put({"type": "tool_call", "id": "t1"})
                w.out_q.put({"type": "answer", "text": "what I did"})
                sess = self._session(w)
                self.assertEqual(sess.pool.bus.attached(), 0)
                sess.pool.bus.pump_once()
                self.assertTrue(w.out_q.empty())
                lines, _truncated = transcript_log.read_since(sid, 0)
                self.assertEqual([e["type"] for e in lines], ["tool_call", "answer"])


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


class HandshakeIsNotBlockedTests(unittest.IsolatedAsyncioTestCase):
    """Nothing may come between a client connecting and its first message being read.

    The job scan walks every session's job directories, and a job directory is never
    swept however old — so on a long-lived workspace it is unbounded file work. Awaited
    inside the handshake it sat between the connection and the message loop: the chat
    came up, said it was ready, and the query was never read — which reads from a chat
    window as "MIMIR is frozen and makes no tool call".
    """

    async def test_a_scan_that_never_finishes_does_not_stop_the_first_message(self):
        w = _bare_worker()
        w.model = "m"
        w.is_busy = lambda: False
        w.get_context_mode = lambda: "full"
        w.get_enforcement = lambda: "light"
        w.get_approval_mode = lambda: "manual"
        w.get_thinking_profile = lambda: {}
        w.get_temperature_state = lambda: {"supported": True, "value": None}
        w._agent = object()
        sess = SessionFencingTests._session(self, w)
        sess._purge_empty_sessions = lambda: None
        sess._send_toggles = _noop_async
        sess._send_sessions_list = _noop_async
        sess._create_new_session = _noop_async
        sess._resend_parked_prompt = _noop_async
        sess._send_served_models = _noop_async
        sess._send_replay = _noop_async
        sess._drain_loop = _noop_async
        sess._summary_task = None
        sess.store = _NoSessions()

        started = asyncio.Event()

        async def _never_finishes():
            started.set()
            await asyncio.Event().wait()

        sess._report_ended_jobs = _never_finishes

        handled: list[str] = []

        async def _handle(raw):
            handled.append(raw)

        sess._handle = _handle
        sess.ws = _OneMessageWS('{"type": "query", "text": "hello"}')

        await asyncio.wait_for(sess.run(), 5)
        self.assertTrue(started.is_set(), "the scan never ran at all")
        self.assertEqual(handled, ['{"type": "query", "text": "hello"}'],
                         "the first message was never read")


class _OneMessageWS(_FakeWS):
    """A socket that yields one message and then ends, like a client that sent one."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def __aiter__(self):
        async def _gen():
            yield self._message
        return _gen()


async def _noop_async(*_args, **_kwargs):
    return None


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
        # The gap this guards: the worker finishes starting and queues its one
        # announcement, but it may have done so before this socket subscribed — and
        # nothing re-emits it, so the chat would sit on "starting the agent" for the
        # life of the connection with a working agent behind it. Asking the worker
        # directly, after the handshake, is what closes it.
        original = sess._purge_empty_sessions

        def _ready_then_setup():
            w._agent = object()
            original()

        sess._purge_empty_sessions = _ready_then_setup

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
