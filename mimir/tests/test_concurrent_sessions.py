"""Several conversations working at once, and answers reaching the right one.

The property the whole change exists for: leaving a conversation leaves its turn
running. What that costs is that "the running turn", "the parked card" and "the answer"
stop being questions with one answer — and every one of them, got wrong, fails silently
rather than loudly.

The sharpest is the answer to a card. Two conversations can be parked at the same moment,
and one may be asking while the user reads another. Routing an answer to "the" agent
would settle a question a different conversation asked, with the user's approval attached
to a tool call they never saw. So a card says which conversation raised it, the answer
carries that back, and anything unattributed is dropped rather than guessed at.

Pure-Python: stub workers and a stub pool, because what is under test is the routing.
"""
from __future__ import annotations

import json
import queue as _queue
import unittest

from mimir.client.ui.ws.ws_session import _Session
from mimir.client.ui.ws.ws_worker import _AgentWorker


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _worker(session_id: str, title: str = "") -> _AgentWorker:
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w._approval_q = _queue.Queue()
    w._question_q = _queue.Queue()
    w._steer_q = _queue.Queue()
    w.session_id = session_id
    w.session_title = title
    w.active_session_id = None
    w._query_session_id = None
    w._pending_prompt = None
    w._pending_questions = None
    w.answered: list = []
    w.resolve_approval = lambda choice, files=None: w.answered.append(("approval", choice))
    w.resolve_question = (
        lambda answers, prompt_id=None: w.answered.append(("question", answers))
    )
    return w


class _ManyPool:
    """A pool holding several real-ish workers, keyed by conversation."""

    def __init__(self, workers: dict) -> None:
        self._workers = dict(workers)
        self.active_session_id = None
        self.model = "test-model"
        self.cap = 3

    def get(self, session_id):
        return self._workers.get(session_id or "")

    def worker_or_detached(self, session_id):
        return self._workers.get(session_id or "") or next(iter(self._workers.values()))

    def items(self):
        return iter(list(self._workers.items()))

    def is_busy(self, session_id):
        w = self._workers.get(session_id or "")
        return w is not None and w._query_session_id is not None

    def is_parked(self, session_id):
        w = self._workers.get(session_id or "")
        return w is not None and w._pending_prompt is not None

    def queued_position(self, session_id):
        return None

    def set_active(self, session_id):
        self.active_session_id = session_id


def _session(pool: _ManyPool, active: str) -> _Session:
    sess = object.__new__(_Session)
    sess.ws = _FakeWS()
    sess.pool = pool
    sess._active_session_id = active
    sess._stale_prompt_ids = set()
    sess._pending_interaction = None
    sess._display_messages = []
    sess._detached_turns = {}
    sess._submitted_len = 0
    sess._live_rows = {}
    sess._sent_progress = {}
    sess._foreign_logs = {}
    pool.set_active(active)
    return sess


class TwoConversationsParkedAtOnceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.a = _worker("s1", "Install the toolchain")
        self.b = _worker("s2", "Port the solver")
        self.pool = _ManyPool({"s1": self.a, "s2": self.b})
        self.sess = _session(self.pool, active="s1")

    def _park(self, worker: _AgentWorker, card_id: str) -> dict:
        worker._query_session_id = worker.session_id
        worker._emit_prompt({"type": "approval", "id": card_id, "tool": "bash_run"})
        return worker.out_q.get_nowait()

    def test_each_card_names_the_conversation_that_raised_it(self) -> None:
        first = self._park(self.a, "card-a")
        second = self._park(self.b, "card-b")
        self.assertEqual(first["session_id"], "s1")
        self.assertEqual(first["session_title"], "Install the toolchain")
        self.assertEqual(second["session_id"], "s2")
        self.assertEqual(second["session_title"], "Port the solver")

    async def test_an_answer_reaches_only_the_conversation_that_asked(self) -> None:
        self._park(self.a, "card-a")
        self._park(self.b, "card-b")
        await self.sess._handle_approval_response(
            {"id": "card-b", "choice": "y", "session_id": "s2"})
        self.assertEqual(self.b.answered, [("approval", "y")])
        self.assertEqual(self.a.answered, [], "answered a question another conversation asked")

    async def test_an_answer_naming_no_conversation_is_dropped(self) -> None:
        """Never defaulted to whoever is on screen: that is exactly the mistake the
        attribution exists to prevent, and it attaches the user's approval to a call
        they never saw."""
        self._park(self.a, "card-a")
        await self.sess._handle_approval_response({"id": "card-a", "choice": "y"})
        self.assertEqual(self.a.answered, [])

    async def test_an_answer_for_an_unknown_conversation_is_dropped(self) -> None:
        self._park(self.a, "card-a")
        await self.sess._handle_approval_response(
            {"id": "card-a", "choice": "y", "session_id": "never-existed"})
        self.assertEqual(self.a.answered, [])

    async def test_a_question_answer_routes_the_same_way(self) -> None:
        self.b._query_session_id = "s2"
        self.b._emit_prompt({"type": "user_question", "id": "q1", "questions": []},
                            questions=[])
        await self.sess._handle_user_question_response(
            {"id": "q1", "answers": [{"selected": ["yes"]}], "session_id": "s2"})
        self.assertEqual(self.b.answered, [("question", [{"selected": ["yes"]}])])
        self.assertEqual(self.a.answered, [])

    async def test_every_parked_card_comes_back_on_reconnect(self) -> None:
        """One left out is a conversation parked for ever: the wait has no timeout."""
        self._park(self.a, "card-a")
        self._park(self.b, "card-b")
        self.a.out_q.get_nowait() if not self.a.out_q.empty() else None
        await self.sess._resend_parked_prompt()
        ids = {m["id"] for m in self.sess.ws.sent}
        self.assertEqual(ids, {"card-a", "card-b"})


class WhatIsRunningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.a = _worker("s1")
        self.b = _worker("s2")
        self.pool = _ManyPool({"s1": self.a, "s2": self.b})
        self.sess = _session(self.pool, active="s1")

    def test_busy_is_asked_of_the_conversation_on_screen(self) -> None:
        self.b._query_session_id = "s2"          # the other one is working
        self.assertFalse(self.sess._running_turn_is_ours())
        self.a._query_session_id = "s1"
        self.assertTrue(self.sess._running_turn_is_ours())

    def test_the_panel_says_which_conversations_are_live(self) -> None:
        self.a._query_session_id = "s1"
        self.b._pending_prompt = {"id": "card-b"}
        self.assertEqual(self.sess._session_activity("s1"),
                         {"running": True, "parked": False, "queued": False})
        self.assertEqual(self.sess._session_activity("s2"),
                         {"running": False, "parked": True, "queued": False})

    def test_an_idle_conversations_queue_is_cleared_and_a_busy_ones_is_not(self) -> None:
        """A turn whose socket dropped mid-run is still working, and what is queued for
        it is its own output waiting for someone to read it."""
        self.a._query_session_id = "s1"          # busy
        self.a.out_q.put({"type": "output", "text": "mid-run"})
        self.b.out_q.put({"type": "output", "text": "debris"})
        self.sess._drop_stale_events()
        self.assertFalse(self.a.out_q.empty(), "threw away a running turn's output")
        self.assertTrue(self.b.out_q.empty())


class DeletingAConversationTests(unittest.IsolatedAsyncioTestCase):
    """Deleting closes the agent, so a turn in flight is cut short — and said first."""

    def _session(self) -> _Session:
        self.a = _worker("s1")
        pool = _ManyPool({"s1": self.a})
        sess = _session(pool, active="s2")
        sess._delete_refused = set()
        return sess

    def test_a_turn_in_progress_is_reported(self) -> None:
        sess = self._session()
        self.a._query_session_id = "s1"
        self.assertEqual(sess._live_work_of("s1"), ["a turn in progress"])

    def test_a_turn_waiting_on_the_user_is_reported_as_such(self) -> None:
        """Parked is work waiting to continue, not work that has stopped."""
        sess = self._session()
        self.a._query_session_id = "s1"
        self.a._pending_prompt = {"id": "card-a"}
        self.assertEqual(sess._live_work_of("s1"), ["a turn waiting for your answer"])

    def test_an_idle_conversation_has_nothing_to_report(self) -> None:
        sess = self._session()
        self.assertEqual(sess._live_work_of("s1"), [])

    async def test_the_first_delete_of_a_working_conversation_is_refused(self) -> None:
        sess = self._session()
        self.a._query_session_id = "s1"
        sess.store = type("S", (), {"delete_session": lambda self, sid: 1 / 0})()
        sess._send_sessions_list = lambda: _noop()
        await sess._handle_delete_session({"session_id": "s1"})
        self.assertIn("still has work running", sess.ws.sent[0]["text"])
        # And asking again goes through: the warning is to be read, not to make the
        # conversation undeletable.
        self.assertIn("s1", sess._delete_refused)


async def _noop() -> None:
    pass


class LeavingAConversationTests(unittest.IsolatedAsyncioTestCase):
    def test_divertible_rows_are_kept_per_conversation(self) -> None:
        """Two conversations can each have a blocking run; a divert must reach its own."""
        pool = _ManyPool({"s1": _worker("s1"), "s2": _worker("s2")})
        sess = _session(pool, active="s1")
        sess._rows_for("s1")["call-1"] = "bash_run"
        sess._rows_for("s2")["call-2"] = "bash_run"
        self.assertEqual(sess._rows_for("s1"), {"call-1": "bash_run"})
        self.assertEqual(sess._rows_for("s2"), {"call-2": "bash_run"})
        sess._forget_row("call-1", "s1")
        self.assertEqual(sess._rows_for("s1"), {})
        self.assertEqual(sess._rows_for("s2"), {"call-2": "bash_run"},
                         "clearing one conversation's rows cleared another's")


if __name__ == "__main__":
    unittest.main()
