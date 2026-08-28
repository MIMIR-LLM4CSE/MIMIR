"""One worker serves every session — events and prompts must not cross over.

A turn parked on a plan-approval prompt used to survive a session switch: its
card stayed on screen, its answer resolved the old turn, and its `open_editor`
opened the *other* session's plan. These tests pin the three seams that fence a
turn to the session it started in.
"""
import queue as _queue
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.client.ui.ws.ws_session import _Session


def _bare_worker() -> _AgentWorker:
    """A worker with only the queue/state fields the tests touch (no agent, no loop)."""
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w._approval_q = _queue.Queue()
    w._continue_q = _queue.Queue()
    w._question_q = _queue.Queue()
    w._steer_q = _queue.Queue()
    w.active_session_id = None
    w._query_session_id = None
    return w


class WorkerStampTests(unittest.TestCase):
    def test_drain_stamps_events_with_the_running_query_session(self):
        w = _bare_worker()
        w._query_session_id = "s1"
        w.out_q.put({"type": "open_editor", "path": "/plans/a.md"})
        self.assertEqual(w.drain()[0]["session_id"], "s1")

    def test_drain_leaves_an_explicit_stamp_alone(self):
        w = _bare_worker()
        w._query_session_id = "s1"
        w.out_q.put({"type": "output", "text": "x", "session_id": "s2"})
        self.assertEqual(w.drain()[0]["session_id"], "s2")

    def test_events_outside_a_query_are_unstamped(self):
        w = _bare_worker()
        w.out_q.put({"type": "output", "text": "x"})
        self.assertIsNone(w.drain()[0]["session_id"])

    def test_flush_prompts_drops_every_pending_answer(self):
        w = _bare_worker()
        w._approval_q.put({"choice": "y"})
        w._continue_q.put({"choice": "y"})
        w._question_q.put({"answers": [{"selected": ["Accept & start"]}]})
        w._steer_q.put("hurry up")
        w.flush_prompts()
        for q in (w._approval_q, w._continue_q, w._question_q, w._steer_q):
            self.assertTrue(q.empty())


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


class _FakeStore:
    def __init__(self, ids):
        self._ids = set(ids)

    def session_exists(self, sid):
        return sid in self._ids


class SessionFencingTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, worker, active="s1"):
        sess = object.__new__(_Session)
        sess.ws = _FakeWS()
        sess.worker = worker
        sess.store = _FakeStore(["s1", "s2"])
        sess._active_session_id = active
        sess._display_messages = []
        return sess

    def test_foreign_events_are_dropped_and_own_events_kept(self):
        sess = self._session(_bare_worker())
        self.assertTrue(sess._is_foreign_event({"type": "open_editor", "session_id": "s2"}))
        self.assertFalse(sess._is_foreign_event({"type": "open_editor", "session_id": "s1"}))
        self.assertFalse(sess._is_foreign_event({"type": "output"}))  # unstamped

    async def test_idle_worker_is_not_cancelled(self):
        w = _bare_worker()
        w.is_busy = lambda: False
        w.cancel = lambda: self.fail("idle worker must not be cancelled")
        self.assertFalse(await self._session(w)._abandon_running_turn())

    async def test_busy_worker_is_cancelled_and_its_prompts_flushed(self):
        w = _bare_worker()
        calls = []
        busy = [True]
        w.is_busy = lambda: busy[0]
        w.cancel = lambda: (calls.append("cancel"), busy.__setitem__(0, False), True)[2]
        w.flush_prompts = lambda: calls.append("flush")
        self.assertTrue(await self._session(w)._abandon_running_turn())
        self.assertEqual(calls, ["cancel", "flush"])

    async def test_switch_cancels_before_the_active_session_pointer_moves(self):
        """Order matters: a late tool call must never write into the new session."""
        w = _bare_worker()
        order = []
        busy = [True]
        w.is_busy = lambda: busy[0]
        w.cancel = lambda: (order.append("cancel"), busy.__setitem__(0, False), True)[2]
        w.flush_prompts = lambda: None
        sess = self._session(w)
        sess._autosave_session = lambda msgs: None
        sess._send_sessions_list = _noop_async
        sess._notify_turn_abandoned = _noop_async

        async def _load(sid):
            order.append("load")

        sess._load_session = _load
        await sess._handle_switch_session({"session_id": "s2"})
        self.assertEqual(order, ["cancel", "load"])


async def _noop_async(*args, **kwargs):
    return None


if __name__ == "__main__":
    unittest.main()
