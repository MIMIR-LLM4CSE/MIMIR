"""The whole chain an event travels, with the real objects.

Every link was tested on its own — the pump, the replay watermark, the committer, the
approval wait — and that is precisely how three separate faults reached a running
install. Each piece was right; what broke was the joins between them, and a chat window
cannot tell those apart: the pump never starting, the pump dying, a watermark swallowing
what it delivers, and the agent emitting nothing all look like a conversation that went
quiet.

So this test is deliberately coarse. It builds a real ``_Session`` over a real
``_EventBus`` and a real worker's queue, runs the handshake, sends a query, and asserts
that what the engine emitted arrived. Coarse on purpose: the joins are what a chat
window cannot see.
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
    # A stand-in agent with the surface the worker actually calls on it. `_approvals`
    # is what the policy engine reads afresh at each gate, so setting the mode is a
    # rebind and nothing has to be queued.
    w._agent = types.SimpleNamespace(
        _cancel_flag=threading.Event(), _deferred_prompts=None, _deferred_turn=None,
        context_mode="full", non_interactive=False,
        set_approval_mode=lambda mode: setattr(w, "_approval_mode", mode),
        toggles_state=lambda: {"servers": [], "skills": [], "nudges": []})
    w._approval_mode = "manual"
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
    w.get_approval_mode = lambda: w._approval_mode
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


class ProseSurvivesAnAbsenceTests(_ChainCase):
    """Coming back shows what the agent *said*, not only what it called.

    The deltas are not journaled — hundreds per turn — so without an aggregate a turn
    read back after an absence keeps only its tool rows and its final answer, having
    lost everything the agent said between its tools. That is most of what makes a turn
    legible. Each block is recorded whole instead, replay-only: a connected client has
    already had the deltas.
    """

    def _turn_with_prose(self) -> list[dict]:
        # Not literal events: the worker builds these from the token callback. The
        # fixture stands in for that, in the order the real one produces.
        return [
            {"type": "assistant_text", "text": "First I will look at the servers.\n"},
            {"type": "tool_call", "id": "c1", "tool": "bash_run"},
            {"type": "tool_result", "id": "c1", "ok": True},
            {"type": "assistant_text", "text": "Nineteen of them; now the client.\n"},
            {"type": "tool_call", "id": "c2", "tool": "read_file"},
            {"type": "tool_result", "id": "c2", "ok": True},
            {"type": "answer", "text": "done", "cancelled": False,
             "_full": None, "_turn_start": None, "_submitted_len": 0,
             "_context_mode": "full"},
        ]

    async def test_the_prose_is_recorded_in_order_with_the_tools(self):
        self._stored_session("s1")
        worker = _worker("s1", turn=self._turn_with_prose())
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "look around"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        logged, _truncated = transcript_log.read_since("s1", 0)
        kinds = [e["type"] for e in logged if e["type"] != "query"]
        self.assertEqual(
            kinds,
            ["assistant_text", "tool_call", "tool_result",
             "assistant_text", "tool_call", "tool_result", "answer"],
            "the record does not interleave what was said with what was called")

    async def test_it_is_not_sent_to_a_client_that_already_had_the_deltas(self):
        # Sending the aggregate as well would print the paragraph twice.
        self._stored_session("s1")
        worker = _worker("s1", turn=self._turn_with_prose())
        pool = FakePool(worker, active="s1")
        ws = _WS([json.dumps({"type": "query", "text": "look around"})])
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        self.assertNotIn("assistant_text", self._kinds(ws))
        self.assertIn("tool_call", self._kinds(ws))

    async def test_a_reconnect_replays_the_prose(self):
        # The property the whole thing is for: leave, come back, and the turn reads as
        # it would have been watched.
        self._stored_session("s1")
        worker = _worker("s1", turn=self._turn_with_prose())
        pool = FakePool(worker, active="s1")
        first = _WS([json.dumps({"type": "query", "text": "look around"})])
        await self._run(_Session(first, pool), pool, first)

        second = _WS([], linger=0.3)
        await self._run(_Session(second, pool), pool, second)

        frames = [m for m in second.sent if m.get("type") == "replay"]
        self.assertTrue(frames, "nothing was replayed")
        replayed = [e["type"] for f in frames for e in f["events"]]
        self.assertEqual(replayed.count("assistant_text"), 2,
                         "the agent's prose did not come back")
        self.assertIn("tool_call", replayed)

    async def test_a_blank_block_is_not_recorded(self):
        # Whitespace between a tool result and the next call is not something the agent
        # said, and recording it puts a blank bubble in every replayed turn. Asserted on
        # the worker's own rule rather than through a turn, because the fixture cannot
        # honour a rule it does not contain.
        worker = _worker("s1")
        self.assertFalse(worker.emit_assistant_text("   \n"))
        self.assertFalse(worker.emit_assistant_text(""))
        self.assertTrue(worker.emit_assistant_text("something said"))
        emitted = []
        while not worker.out_q.empty():
            emitted.append(worker.out_q.get_nowait())
        self.assertEqual([e["text"] for e in emitted], ["something said"])


class ComingBackToARunningTurnTests(_ChainCase):
    """A reattach is a window opening onto a run that never stopped.

    Two things follow, and both are the user's own statement of what they want: the
    conversation comes back in the mode it was left running under, and a turn that was
    still working carries on — the window is what went away, not the run.
    """

    def setUp(self) -> None:
        super().setUp()
        from mimir.client.ui.ws import server_registry
        patcher = mock.patch.object(server_registry, "_MIMIR_DIR_WS", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = server_registry

    def _detached_run(self, autonomy: str, session: str = "s1") -> None:
        """A registry entry describing a detached run, as `detach` leaves it.

        The claim is per conversation: each one is left under its own level, and the
        process survives for as long as any of them claims it.
        """
        self.registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        self.registry.claim([session], autonomy)

    async def test_the_conversation_comes_back_under_the_level_it_was_left_on(self):
        self._stored_session("s1")
        self._detached_run("auto")
        worker = _worker("s1")
        applied: list[tuple] = []
        pool = FakePool(worker, active="s1")
        pool.apply_setting = lambda name, *a: applied.append((name, a)) or []
        ws = _WS([], linger=0.3)
        sess = _Session(ws, pool)

        await self._run(sess, pool, ws)

        self.assertIn(("set_approval_mode", ("auto",)), applied,
                      "the run's own level was not restored")
        modes = [m for m in ws.sent if m.get("type") == "approval_mode"]
        self.assertTrue(modes, "the panel was never told which level it is on")
        self.assertEqual(modes[-1]["mode"], "auto")

    async def test_a_rebuilt_worker_comes_up_on_that_level_too(self):
        # Recorded pool-wide as well as applied: a worker built after the reattach
        # would otherwise start on the default and park at its next sensitive call,
        # silently dropping a run from auto to manual.
        self._stored_session("s1")
        self._detached_run("auto_all")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([], linger=0.3)

        await self._run(_Session(ws, pool), pool, ws)

        self.assertEqual(worker.get_approval_mode(), "auto_all")

    async def test_a_server_that_is_not_detached_is_left_alone(self):
        # An ordinary connect decides nothing about the approval mode.
        self._stored_session("s1")
        self.registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        worker = _worker("s1")
        applied: list[tuple] = []
        pool = FakePool(worker, active="s1")
        pool.apply_setting = lambda name, *a: applied.append((name, a)) or []
        ws = _WS([], linger=0.3)

        await self._run(_Session(ws, pool), pool, ws)

        self.assertEqual([a for a in applied if a[0] == "set_approval_mode"], [])

    async def test_a_turn_still_in_flight_is_reported_as_running(self):
        # So the chat comes back busy, with a stop button that stops something — and,
        # not cosmetically, knowing a turn is open so the answer that ends it hands the
        # finished transcript back.
        self._stored_session("s1")
        worker = _worker("s1")
        worker._query_session_id = "s1"            # a turn of ours, mid-flight
        pool = FakePool(worker, active="s1")
        ws = _WS([], linger=0.3)

        await self._run(_Session(ws, pool), pool, ws)

        loaded = [m for m in ws.sent if m.get("type") == "session_loaded"]
        self.assertTrue(loaded)
        self.assertTrue(loaded[-1]["turn_running"],
                        "the window came back without knowing its turn was still going")

    async def test_an_idle_conversation_is_not_reported_as_running(self):
        self._stored_session("s1")
        worker = _worker("s1")
        pool = FakePool(worker, active="s1")
        ws = _WS([], linger=0.3)

        await self._run(_Session(ws, pool), pool, ws)

        loaded = [m for m in ws.sent if m.get("type") == "session_loaded"]
        self.assertFalse(loaded[-1]["turn_running"])

    async def test_the_turn_keeps_producing_across_the_reattach(self):
        # The property itself: the window went away, the run did not.
        self._stored_session("s1")
        worker = _worker("s1", turn=[])
        pool = FakePool(worker, active="s1")

        first = _WS([json.dumps({"type": "query", "text": "a long one"})], linger=0.2)
        await self._run(_Session(first, pool), pool, first)

        # Mid-turn output, produced while nothing is attached.
        worker._query_session_id = "s1"
        worker.out_q.put({"type": "status", "text": "  → still going\n",
                          "session_id": "s1"})

        second = _WS([], linger=0.4)
        await self._run(_Session(second, pool), pool, second)

        replayed = [e["type"] for m in second.sent if m.get("type") == "replay"
                    for e in m["events"]]
        self.assertIn("status", replayed,
                      "what the turn produced while away did not come back")


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
