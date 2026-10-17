"""A connection forwards its share of the stream, and costs nothing while idle.

The loop it replaces slept 5 ms between looks at its queue and ``stat``-ed the checklist
file each time round, on every connection, for the life of the socket. These pin the
replacement: it waits on the subscription, refreshes only while a turn on screen is
live, and does per event type what the old loop did inline.

Pure-Python (no agent, no servers): runs on x86 and ARM.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws.connection import router as event_router
from mimir.client.ui.ws.connection.router import EventRouter
from mimir.tests._fake_pool import FakePool
from mimir.tests._session import bare_session


class _Worker:
    model = "m"

    def __init__(self) -> None:
        self.batch: list[dict] = []
        self.todos: list[dict] = []
        self._query_session_id = None

    def is_busy(self) -> bool:
        return self._query_session_id is not None

    def _build_batch_status(self) -> list[dict]:
        return list(self.batch)

    def _load_todos(self) -> list[dict]:
        return list(self.todos)


class _RouterCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.worker = _Worker()
        self.pool = FakePool(self.worker, active="s1")
        self.sess = bare_session(self.pool)
        self.sess._sub = self.pool.bus.subscribe()
        self.router = EventRouter(self.sess)

    def _sent(self, kind: str) -> list[dict]:
        return [m for m in self.sess.ws.sent if m.get("type") == kind]


class WaitingTests(_RouterCase):
    async def test_an_idle_connection_waits_without_a_timeout(self) -> None:
        # Nothing live on screen: the next look is the next event, however long.
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(self.router._next(self.sess._sub.queue), 0.2)

    async def test_a_live_turn_wakes_it_once_a_tick_for_its_refreshes(self) -> None:
        self.worker._query_session_id = "s1"
        with mock.patch.object(event_router, "_TICK", 0.05):
            self.assertIsNone(await asyncio.wait_for(
                self.router._next(self.sess._sub.queue), 1.0))

    async def test_an_event_is_returned_as_soon_as_it_is_there(self) -> None:
        self.sess._sub.offer({"type": "status", "text": "x", "session_id": "s1"})
        ev, _extras = await asyncio.wait_for(self.router._next(self.sess._sub.queue), 1.0)
        self.assertEqual(ev["type"], "status")


class PerEventTests(_RouterCase):
    async def test_a_divertible_row_is_remembered_until_its_result(self) -> None:
        await self.router.route({"type": "tool_call", "id": "c1", "name": "bash",
                                 "divertible": True, "session_id": "s1"}, {})
        self.assertEqual(self.sess._rows_for("s1"), {"c1": "bash"})
        await self.router.route({"type": "tool_result", "id": "c1", "session_id": "s1"}, {})
        self.assertEqual(self.sess._rows_for("s1"), {})

    async def test_a_diff_pushes_the_review_bar_only_when_there_is_one(self) -> None:
        await self.router.route({"type": "diff", "session_id": "s1"}, {})
        self.assertEqual(self._sent("batch_status"), [])
        self.worker.batch = [{"file": "a.py"}]
        await self.router.route({"type": "diff", "session_id": "s1"}, {})
        self.assertEqual(self._sent("batch_status")[-1]["files"], [{"file": "a.py"}])

    async def test_another_conversations_event_is_withheld_and_counted(self) -> None:
        await self.router.route({"type": "tool_call", "id": "c1", "session_id": "s2"}, {})
        self.assertEqual(self._sent("tool_call"), [])
        self.assertEqual(self.sess._foreign_withheld, {"s2": 1})

    async def test_a_wake_says_whether_it_started_a_turn_here(self) -> None:
        self.sess._send_sessions_list = mock.AsyncMock()
        await self.router.route({"type": "job_complete", "job_key": "j1",
                                 "session_id": "s1"}, {"_started": True})
        await self.router.route({"type": "job_complete", "job_key": "j2",
                                 "session_id": "s2"}, {"_started": True})
        flags = [m["resumes_active_session"] for m in self._sent("job_complete")]
        # A turn started in another conversation does not make this chat busy.
        self.assertEqual(flags, [True, False])

    async def test_the_checklist_is_sent_when_its_file_changed_not_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "todo_list.md")
            self.worker.todos = [{"text": "step", "done": False}]
            with mock.patch.object(event_router, "_todo_file_for_session",
                                   lambda _sid: path):
                with open(path, "w") as fh:
                    fh.write("- [ ] step\n")
                await self.router.route({"type": "tool_result", "id": "c1",
                                         "session_id": "s1"}, {})
                await self.router.route({"type": "tool_result", "id": "c2",
                                         "session_id": "s1"}, {})
        self.assertEqual(len(self._sent("todo")), 1)


if __name__ == "__main__":
    unittest.main()
