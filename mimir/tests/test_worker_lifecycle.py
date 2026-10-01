"""A worker belongs to one conversation, and can be let go.

Two properties the agent pool depends on, pinned here because neither is visible from
the pool's own code.

**It knows its conversation.** A worker with a session of its own resolves that session's
checklist and stamps its output with it, whatever the user happens to be looking at. A
worker shared between conversations could only stream one of them anywhere the user can
see, which is what would force leaving a conversation to stop its turn.

**It can be closed.** ``shutdown`` ends the query loop and nothing else; the agent's
``exit_stack`` holds ~19 MCP server subprocesses, and a worker released when its
conversation goes quiet must close it or every release strands a full set of servers.
``aclose`` only asks: the close belongs to the task that opened those servers, because
``stdio_client`` anchors its anyio cancel scope there and exiting it from any other task
raises and leaves the stack half-unwound — streams closed, subprocess alive. So the
sentinel is the mechanism, and the query loop's own ``finally`` does the closing.

Pure-Python (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import asyncio
import queue as _queue
import threading
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker


class _FakeAgent:
    """Records that it was closed, and on which thread."""

    def __init__(self, hang: bool = False) -> None:
        self.closed = False
        self.closed_on: int | None = None
        self._hang = hang
        self._cancel_flag = threading.Event()

    async def cleanup(self) -> None:
        if self._hang:
            await asyncio.sleep(3600)
        self.closed = True
        self.closed_on = threading.get_ident()


def _worker_with_loop(agent: object | None) -> tuple[_AgentWorker, threading.Thread, int]:
    """A worker whose loop really runs on its own thread, without building an agent."""
    w = object.__new__(_AgentWorker)
    w.session_id = "s1"
    w.active_session_id = None
    w._query_session_id = None
    w._agent = agent
    w._bg_jobs = {}
    w._query_q = _queue.Queue()
    w._query_event = threading.Event()
    w._closed = threading.Event()
    w._current_task = None
    w.out_q = _queue.Queue()

    loop = asyncio.new_event_loop()
    started = threading.Event()
    ident: dict[str, int] = {}

    async def _live() -> None:
        """The shape of _AgentWorker._live: serve until the sentinel, then close.

        One task for both halves, which is the property under test: the close runs where
        the servers were opened.
        """
        try:
            while True:
                try:
                    if w._query_q.get_nowait() is None:
                        return
                except _queue.Empty:
                    await asyncio.sleep(0.01)
        finally:
            await w._close_agent()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        ident["id"] = threading.get_ident()
        loop.call_soon(started.set)
        try:
            loop.run_until_complete(_live())
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait(5)
    w._loop = loop
    w._thread = thread
    return w, thread, ident["id"]


class OwnSessionTests(unittest.TestCase):
    def _bare(self, session_id=None, active=None) -> _AgentWorker:
        w = object.__new__(_AgentWorker)
        w.session_id = session_id
        w.active_session_id = active
        return w

    def test_its_own_session_wins_over_the_one_on_screen(self):
        """The whole point: the user looking elsewhere changes nothing for this worker."""
        self.assertEqual(self._bare("mine", active="someone-else")._own_session(), "mine")

    def test_a_worker_without_one_falls_back_to_the_screen(self):
        """How a worker with no conversation of its own resolves: tests, standalone use."""
        self.assertEqual(self._bare(None, active="on-screen")._own_session(), "on-screen")

    def test_neither_is_no_session(self):
        self.assertIsNone(self._bare(None, None)._own_session())


class ACloseTests(unittest.TestCase):
    def test_the_agents_servers_are_closed(self):
        agent = _FakeAgent()
        w, thread, _ = _worker_with_loop(agent)
        w.aclose()
        self.assertTrue(agent.closed, "MCP servers were left running")

    def test_the_close_runs_on_the_workers_own_thread(self):
        """Closing an anyio context from another task leaks the subprocess."""
        agent = _FakeAgent()
        w, _thread, loop_thread_id = _worker_with_loop(agent)
        w.aclose()
        self.assertEqual(agent.closed_on, loop_thread_id)

    def test_a_wedged_server_does_not_hold_the_caller_for_ever(self):
        """A server stuck in its own shutdown must not hold the pool.

        And the close it is still attempting must survive the way out, so the caller
        returns and the thread is left running it rather than being joined out from under
        it — the loop closing would abandon the very subprocess the close waits for.
        """
        import time
        agent = _FakeAgent(hang=True)
        w, thread, _ = _worker_with_loop(agent)
        started = time.monotonic()
        with self.assertLogs("mimir.client.ui.ws.ws_worker", "WARNING") as log:
            w.aclose(timeout=0.3)
        waited = time.monotonic() - started
        self.assertLess(waited, 2.0, f"held the caller for {waited:.1f}s")
        self.assertFalse(agent.closed)
        self.assertIn("did not close", "\n".join(log.output))
        self.assertTrue(thread.is_alive(), "stopped the thread mid-close")

    def test_the_thread_is_gone_after_a_clean_close(self):
        agent = _FakeAgent()
        w, thread, _ = _worker_with_loop(agent)
        w.aclose()
        self.assertFalse(thread.is_alive())

    def test_a_worker_that_never_built_an_agent_closes_cleanly(self):
        """Setup can fail before the agent exists; the pool still has to let it go."""
        w, _thread, _ = _worker_with_loop(None)
        w.aclose()   # must not raise

    def test_the_query_loop_is_told_to_stop(self):
        """Proven by the loop returning: it only does so on the sentinel."""
        agent = _FakeAgent()
        w, thread, _ = _worker_with_loop(agent)
        w.aclose()
        self.assertFalse(thread.is_alive(), "the query loop never saw the sentinel")
        self.assertTrue(w._query_q.empty(), "the sentinel was queued but not consumed")


if __name__ == "__main__":
    unittest.main()
