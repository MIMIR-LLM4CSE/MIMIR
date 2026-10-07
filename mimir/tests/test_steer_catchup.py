"""Steering typed into the last step of a run still gets answered.

A steer is taken in at a step boundary. The step that writes the final answer has none
after it, so a message typed while that answer streams reaches the queue once the loop
has stopped draining — and before this it stayed there: the run ended, nothing read it,
and the webview kept showing it as "queued" for a turn that was over.

Two halves, pinned separately because they fail apart:

**The worker knows.** Its steer queue is the ground truth for what the loop never read,
so the leftovers leave with the answer — unless the turn was cancelled, where the
instruction aimed at it is abandoned with it.

**The session answers it.** The leftovers become the next turn, the message reads once
in the history, and the client is told so the bubble loses its tag and the chat marks
itself busy for a turn nobody pressed send for.

Pure-Python (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import asyncio
import json
import queue as _queue
import shutil
import tempfile
import threading
import types
import unittest
from unittest import mock

from mimir.client.ui.ws.event_bus import _pop_answer_extras
from mimir.client.ui.ws.ws_worker import _AgentWorker

from mimir.tests._fake_pool import FakePool


class _FakeAgent:
    """Runs a turn that reads no steer, the way a final-answer step does."""

    def __init__(self) -> None:
        self._cancel_flag = threading.Event()
        self._deferred_prompts: list = []
        self._deferred_turn = None
        self._last_full_messages = None
        self._last_turn_start = None
        self.context_mode = "flat"
        self.streaming = False
        self.thinking = False

    async def run(self, **_kw) -> str:
        return "done"


def _worker() -> _AgentWorker:
    """A worker with just the state ``_run_query`` touches."""
    w = object.__new__(_AgentWorker)
    w.session_id = "s1"
    w.active_session_id = None
    w._query_session_id = "s1"
    w._agent = _FakeAgent()
    w._steer_q = _queue.Queue()
    w._defer = set()
    w._preanswer = None
    w._current_task = None
    w._turn_submitted_len = 0
    w.out_q = _queue.Queue()
    w._load_todos = lambda: []
    w._count_tokens = lambda _text: 0
    return w


def _answer(w: _AgentWorker) -> dict:
    """The answer event of one run of *w*."""
    asyncio.run(w._run_query({"text": "do it", "history": []}))
    events = list(w.out_q.queue)
    return next(ev for ev in events if ev.get("type") == "answer")


class UnreadSteerLeavesWithTheAnswerTests(unittest.TestCase):
    def test_a_steer_the_loop_never_read_travels_on_the_answer(self) -> None:
        w = _worker()
        w.submit_steer("stop and use the other file")
        self.assertEqual(_answer(w)["_unconsumed_steer"],
                         ["stop and use the other file"])

    def test_the_queue_is_left_empty_so_it_cannot_bleed_into_the_next_turn(self) -> None:
        """Carried forward instead, it would be injected into whatever runs next."""
        w = _worker()
        w.submit_steer("stop and use the other file")
        _answer(w)
        self.assertTrue(w._steer_q.empty())

    def test_a_turn_that_read_its_steering_carries_nothing(self) -> None:
        w = _worker()
        w.submit_steer("read me")
        w._agent._poll_steer = w._drain_steer_q   # what the loop patches on
        w._agent.run = lambda **_kw: _read_then_answer(w)
        self.assertNotIn("_unconsumed_steer", _answer(w))

    def test_a_cancelled_turn_abandons_its_steering(self) -> None:
        """The instruction was aimed at a turn the user stopped."""
        w = _worker()

        async def _cancelled(**_kw):
            w.submit_steer("too late")
            raise asyncio.CancelledError

        w._agent.run = _cancelled
        ev = _answer(w)
        self.assertNotIn("_unconsumed_steer", ev)
        self.assertTrue(w._steer_q.empty())

    def test_it_reaches_the_session_and_not_the_wire(self) -> None:
        """A private key: the client is told by the session, in its own message."""
        ev = {"type": "answer", "text": "done", "_unconsumed_steer": ["x"]}
        extras = _pop_answer_extras(ev)
        self.assertEqual(extras["_unconsumed_steer"], ["x"])
        self.assertNotIn("_unconsumed_steer", ev)


async def _read_then_answer(w: _AgentWorker) -> str:
    w._agent._poll_steer()
    return "done"


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


class _FakeWorker:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.steered: list[str] = []
        self._query_session_id = "s1"
        self._agent = types.SimpleNamespace(context_mode="flat")
        self.model = "test-model"

    def submit_query(self, text, history, session_id=None) -> None:
        self.submitted.append((text, list(history), session_id))

    def submit_steer(self, text) -> None:
        self.steered.append(text)

    def is_busy(self) -> bool:
        return True

    def watched_job_keys(self) -> list[str]:
        return []


class TheCatchUpTurnTests(unittest.TestCase):
    """What the session does with steering the run handed back unread."""

    def setUp(self) -> None:
        from mimir.client.ui.ws import session_store, transcript_log
        from mimir.client.ui.ws.ws_session import _Session

        self._tmp = tempfile.mkdtemp(prefix="mimir-steer-")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        for target, attr in ((session_store, "STATE_DIR"),
                             (transcript_log, "_MIMIR_DIR_WS")):
            patcher = mock.patch.object(target, attr, self._tmp)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.ws, self.worker = _FakeWS(), _FakeWorker()
        self.session = _Session(self.ws, FakePool(self.worker))
        meta = self.session.store.new_session()
        self.session.store.save_session(meta)
        self.sid = meta.id
        self.session._active_session_id = self.sid
        self.session.history = [{"role": "user", "content": "do it"}]
        self.session.history_full = list(self.session.history)
        self.session._display_messages = []
        self.session._running_turn_is_ours = lambda: True

    def _steer(self, text: str) -> None:
        asyncio.run(self.session._handle_steer({"text": text}))

    def _flush(self, texts: list[str]) -> None:
        asyncio.run(self.session._flush_unconsumed_steer(self.sid, texts))

    def test_the_unread_message_starts_the_next_turn(self) -> None:
        self._steer("use the other file")
        self._flush(["use the other file"])
        self.assertEqual(len(self.worker.submitted), 1)
        text, _history, session_id = self.worker.submitted[0]
        self.assertEqual(text, "use the other file")
        self.assertEqual(session_id, self.sid)

    def test_the_message_reads_once_in_the_history(self) -> None:
        """The copy written when it was typed is in the middle; the turn starts at the end."""
        self._steer("use the other file")
        self._flush(["use the other file"])
        _text, history, _sid = self.worker.submitted[0]
        said = [m for m in history if m.get("content") == "use the other file"]
        self.assertEqual(len(said), 1)
        self.assertEqual(history[-1]["content"], "use the other file")

    def test_an_older_identical_turn_is_not_swallowed(self) -> None:
        """Removal is by identity: the same words said before are a different turn."""
        self.session.history.append({"role": "user", "content": "again"})
        self.session.history_full = list(self.session.history)
        self._steer("again")
        self._flush(["again"])
        _text, history, _sid = self.worker.submitted[0]
        self.assertEqual(sum(1 for m in history if m.get("content") == "again"), 2)

    def test_the_client_is_told_the_bubble_was_read_and_a_turn_began(self) -> None:
        self._steer("use the other file")
        self._flush(["use the other file"])
        injected = [m for m in self.ws.sent if m.get("type") == "steer_injected"]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["text"], "use the other file")
        self.assertTrue(injected[0]["starts_turn"])

    def test_several_unread_messages_leave_as_one_turn_but_clear_both_bubbles(self) -> None:
        self._steer("first")
        self._steer("second")
        self._flush(["first", "second"])
        self.assertEqual(len(self.worker.submitted), 1)
        text, _history, _sid = self.worker.submitted[0]
        self.assertIn("first", text)
        self.assertIn("second", text)
        injected = [m for m in self.ws.sent if m.get("type") == "steer_injected"]
        self.assertEqual([m["text"] for m in injected], ["first", "second"])
        self.assertEqual([m["starts_turn"] for m in injected], [False, True])

    def test_a_turn_that_read_its_steering_starts_nothing(self) -> None:
        self._steer("read in time")
        self._flush([])
        self.assertEqual(self.worker.submitted, [])
        self.assertEqual([m for m in self.ws.sent if m.get("type") == "steer_injected"], [])

    def test_the_bookkeeping_does_not_outlive_the_turn(self) -> None:
        """Entries kept past the turn would delete a later turn's history from under it."""
        self._steer("read in time")
        self._flush([])
        self.assertEqual(self.session._live_steer, [])


if __name__ == "__main__":
    unittest.main()
