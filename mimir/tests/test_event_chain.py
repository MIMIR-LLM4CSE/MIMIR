"""The whole chain an event travels, with the real objects.

Every link was tested on its own — the pump, the replay watermark, the committer, the
approval wait — and that is precisely how three separate faults reached a running
install. Each piece was right; what broke was the joins between them, and a chat window
cannot tell those apart: the pump never starting, the pump dying, a watermark swallowing
what it delivers, and the agent emitting nothing all look like a conversation that went
quiet.

So this test is deliberately coarse. It builds a real ``_Session`` over a real
``_EventBus`` and a real worker's queue, runs the handshake, sends a query, and asserts
that what the engine emitted arrived. It is the test that was missing.
"""
import asyncio
import json
import queue as _queue
import tempfile
import threading
import types
import unittest
from unittest import mock

from mimir.client.ui.ws import job_scan, session_store, transcript_log, ws_session
from mimir.client.ui.ws.ws_session import _Session
from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests._fake_pool import FakePool

# What a turn that runs one tool actually puts on the queue, in order.
_TURN = [
    {"type": "status", "text": "  → bash_run\n"},
    {"type": "tool_call", "id": "c1", "tool": "bash_run", "divertible": True},
    {"type": "tool_result", "id": "c1", "ok": True},
    {"type": "answer", "text": "the job is running", "cancelled": False,
     "_full": None, "_turn_start": None, "_submitted_len": 0, "_context_mode": "full"},
]


class _WS:
    """A socket that yields the given messages, then idles long enough for the turn."""

    def __init__(self, messages: list[str], linger: float = 1.0) -> None:
        self.sent: list[dict] = []
        self._messages = list(messages)
        self._linger = linger

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def __aiter__(self):
        async def _gen():
            for message in self._messages:
                yield message
                await asyncio.sleep(0.05)
            await asyncio.sleep(self._linger)
        return _gen()


def _worker(session_id: str, turn: list[dict] | None = None) -> _AgentWorker:
    """A worker with no agent and no loop, whose query emits *turn*."""
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w._approval_q = _queue.Queue()
    w._question_q = _queue.Queue()
    w.session_id = session_id
    w.session_title = "a conversation"
    w._query_session_id = None
    w.active_session_id = session_id
    w.model = "m"
    w._agent = types.SimpleNamespace(
        _cancel_flag=threading.Event(), _deferred_prompts=None, _deferred_turn=None,
        context_mode="full", non_interactive=False)
    w._defer = threading.Event()
    w._preanswer = None
    w._pending_prompt = None
    w._pending_questions = None
    w._expired_prompt_ids = set()
    w._bg_jobs = {}
    w.unattended_since = None
    w.has_deferral = False
    w.is_busy = lambda: w._query_session_id is not None
    w.has_work_pending = lambda: False
    w.agent_ready = lambda: True
    w.pending_prompt = lambda: None
    w.get_context_mode = lambda *a: "full"
    w.get_enforcement = lambda: "light"
    w.get_approval_mode = lambda: "manual"
    w.get_thinking_profile = lambda: {}
    w.get_temperature_state = lambda: {"supported": True, "value": None}
    w._load_todos = lambda: []
    w.export_agent_state = lambda: {"carry_context": {}}
    w.full_history = lambda: None
    w.load_agent_state = lambda _s: None
    w.last_turn_start = lambda: None
    w.toggles_state = lambda: {"servers": [], "skills": [], "nudges": []}
    w.list_resources = lambda: []

    def submit_query(text, history, session_id=None):
        # Stamped with the session the turn was submitted *for*, which is what the real
        # worker does: `_query_session_id` is set from the submission and `drain()`
        # reads it. Leaving the events to fall back on `w.session_id` would make this
        # fixture report a conversation the turn does not belong to — and the drain
        # loop would then correctly refuse to draw them, so the test would be measuring
        # the fixture.
        owner = session_id or w.session_id
        w._query_session_id = owner
        for event in (turn if turn is not None else _TURN):
            w.out_q.put({**event, "session_id": owner})
        w._query_session_id = None

    w.submit_query = submit_query
    return w


class _ChainCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for target, attr in ((transcript_log, "_MIMIR_DIR_WS"),
                             (session_store, "STATE_DIR"),
                             (ws_session, "_MIMIR_DIR_WS"),
                             (job_scan, "_MIMIR_DIR_WS")):
            patcher = mock.patch.object(target, attr, self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        # The summary refresh calls the backend; nothing here has one.
        patcher = mock.patch.object(_Session, "_schedule_summary_refresh",
                                    lambda _self: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stored_session(self, session_id: str) -> None:
        """A conversation already on disk, which is what a reconnect finds."""
        store = session_store.SessionStore()
        session = store.new_session()
        session.id = session_id
        session.title = "a conversation"
        session.display_messages = [{"role": "user", "kind": "text", "text": "hello"}]
        store.save_session(session)

    async def _run(self, sess, pool, ws) -> None:
        """Drive run() with the pump going, as the server does."""
        async def _pump():
            while True:
                pool.bus.pump_once()
                await asyncio.sleep(0.01)

        pumper = asyncio.create_task(_pump())
        try:
            await asyncio.wait_for(sess.run(), 20)
        finally:
            pumper.cancel()

    @staticmethod
    def _kinds(ws) -> list[str]:
        return [m.get("type") for m in ws.sent]


class ATurnReachesTheClientTests(_ChainCase):
    async def test_everything_the_engine_emitted_arrives(self):
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "run something long"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        kinds = self._kinds(ws)
        for expected in ("ready", "session_loaded", "status", "tool_call",
                         "tool_result", "answer"):
            self.assertIn(expected, kinds, f"{expected} never reached the client")

    async def test_nothing_was_swallowed_by_the_watermark(self):
        # The failure that looks like a half-working chat: `token` and `thinking` are
        # never journaled, so they bypass the gate, while everything stamped is
        # filtered. A gate set too high therefore streams text and reasoning and shows
        # no tool call and no answer.
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "run something long"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        self.assertEqual(sess._sub.filtered, 0,
                         "the replay watermark dropped live events")
        self.assertEqual(sess._sub.dropped, 0)

    async def test_the_journal_holds_the_same_turn_the_client_saw(self):
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "run something long"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        logged, _truncated = transcript_log.read_since("s1", 0)
        kinds = [e["type"] for e in logged]
        self.assertEqual(kinds[0], "query", "the client's own line is not recorded")
        for expected in ("status", "tool_call", "tool_result", "answer"):
            self.assertIn(expected, kinds)

    async def test_a_second_turn_also_arrives(self):
        # The watermark advances between turns; a gate that over-shot would silence
        # the second one only.
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "first"}),
                  json.dumps({"type": "query", "text": "second"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        answers = [m for m in ws.sent if m.get("type") == "answer"]
        self.assertEqual(len(answers), 2, "the second turn never reached the client")

    async def test_a_brand_new_conversation_delivers_its_first_turn(self):
        # No session on disk: run() creates one, and its journal starts from nothing.
        # A watermark inherited from a previous conversation would sit above every
        # event this one will ever produce. Asserted on what the client receives rather
        # than on the field, because two separate guards hold this property — the reset
        # in `_create_new_session` and the clamp in `_send_replay` — and what matters is
        # that the turn arrives, not which of them saw to it. The reset has its own
        # test in test_reattach_replay.
        worker = _worker("s1")
        pool = FakePool(worker, active=None)
        ws = _WS([json.dumps({"type": "query", "text": "run something long"})])
        sess = _Session(ws, pool)
        sess._rendered_seq = 4321          # as a previous conversation would leave it

        await self._run(sess, pool, ws)

        kinds = self._kinds(ws)
        for expected in ("status", "tool_call", "tool_result", "answer"):
            self.assertIn(expected, kinds,
                          f"{expected} was silenced in a new conversation")
        self.assertEqual(sess._sub.filtered, 0)


class WhenTheChatGoesQuietTests(_ChainCase):
    """The ways it can, pinned so each is deliberate rather than a surprise."""

    async def test_an_event_of_another_conversation_is_not_drawn_here(self):
        # Deliberate: a turn of a conversation that is not on screen has nowhere to be
        # drawn. It is journaled under its own session instead, which is what makes
        # switching to it show the whole turn. Pinned because it is also the one way a
        # chat can go quiet while the journal fills — and that is worth recognising.
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "run something long"})])
        sess = _Session(ws, pool)

        # The turn stamps itself for another conversation, which is what a background
        # job's wake turn does: it runs in the session that launched the run, not in
        # whichever one is on screen.
        def _submit_elsewhere(text, history, session_id=None):
            for event in _TURN:
                worker.out_q.put({**event, "session_id": "elsewhere"})
        worker.submit_query = _submit_elsewhere

        await self._run(sess, pool, ws)

        self.assertNotIn("tool_call", self._kinds(ws))
        logged, _truncated = transcript_log.read_since("elsewhere", 0)
        self.assertIn("tool_call", [e["type"] for e in logged],
                      "the other conversation's turn was not recorded either")

    async def test_a_pump_that_never_ran_is_visible_in_the_diagnostics(self):
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        rows = {r["label"]: r["detail"] for r in pool.bus.diagnostics()}
        self.assertIn("NOT RUNNING", rows["pump"])

    async def test_the_diagnostics_name_every_link(self):
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "x"}),
                  json.dumps({"type": "command", "text": "/diag"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        reply = [m for m in ws.sent if m.get("type") == "command_output"]
        self.assertTrue(reply, "/diag answered nothing")
        labels = {item["label"] for item in reply[0]["items"]}
        for expected in ("pump", "journaled", "fanned out", "subscriptions",
                         "active session", "rendered watermark"):
            self.assertIn(expected, labels)
        self.assertTrue(any(label.startswith("worker ") for label in labels))


class TheApprovalCardReachesTheClientTests(_ChainCase):
    """A card that does not arrive is a turn parked for ever with nothing on screen."""

    def _parked_worker(self):
        w = _worker("s1", turn=[])
        return w

    async def test_the_card_arrives_and_the_answer_comes_back(self):
        worker = self._parked_worker()
        pool = FakePool(worker, active="s1")
        bus = pool.bus
        sub = bus.subscribe()

        async def _pump():
            while True:
                bus.pump_once()
                await asyncio.sleep(0.01)

        pumper = asyncio.create_task(_pump())
        try:
            await asyncio.sleep(0.05)
            self.assertIsNone(worker.unattended_since,
                              "reported as unattended while a client is subscribed")

            answer: dict = {}

            def _park():
                worker._emit_prompt({"type": "approval", "id": "a1",
                                     "tool": "bash_run"})
                answer["seen"] = worker._await_response(worker._approval_q)

            thread = threading.Thread(target=_park, daemon=True)
            thread.start()
            for _ in range(100):
                await asyncio.sleep(0.02)
                if not sub.queue.empty():
                    break

            delivered = []
            while not sub.queue.empty():
                event, _extras = sub.queue.get_nowait()
                delivered.append(event["type"])
            self.assertIn("approval", delivered, "the card never reached the client")

            worker._approval_q.put({"choice": "y"})
            thread.join(3)
            self.assertEqual(answer.get("seen"), {"choice": "y"})
            self.assertFalse(worker.has_deferral,
                             "set the card aside with a client attached")
        finally:
            pumper.cancel()


if __name__ == "__main__":
    unittest.main()
