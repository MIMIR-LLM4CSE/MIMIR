"""A worker belongs to one conversation, and can be let go.

Two properties the agent pool depends on, pinned here because neither is visible from
the pool's own code.

**It knows its conversation.** One worker used to serve every session, time-multiplexed,
which is why leaving a session had to cancel its turn — a single worker cannot stream two
conversations anywhere the user can see. A worker with a session of its own resolves its
own checklist and stamps its own output, whatever the user is looking at.

**It can be closed.** ``shutdown`` ends the query loop and nothing else; the agent's
``exit_stack``, which holds ~19 MCP server subprocesses, was never closed. Harmless while
one worker lived as long as the process, and the opposite of harmless once workers are
created per conversation and released when one goes quiet: every release would strand a
full set of servers. ``aclose`` is what makes releasing a worker actually free it, and it
must close on the worker's own loop — ``stdio_client`` is an anyio context entered there,
and closing it from another task leaks the subprocess it was meant to reap.

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
    w.out_q = _queue.Queue()

    loop = asyncio.new_event_loop()
    started = threading.Event()
    ident: dict[str, int] = {}

    async def _query_loop() -> None:
        """The shape of _AgentWorker._main: run until the shutdown sentinel arrives."""
        while True:
            try:
                if w._query_q.get_nowait() is None:
                    return
            except _queue.Empty:
                await asyncio.sleep(0.01)

    def _run() -> None:
        asyncio.set_event_loop(loop)
        ident["id"] = threading.get_ident()
        loop.call_soon(started.set)
        try:
            loop.run_until_complete(_query_loop())
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
        """How the single shared worker worked, and how a bare test worker still does."""
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

        And the close it is still attempting must not be destroyed on the way out: the
        shutdown sentinel ends the query loop, whose return closes the event loop — which
        would kill the pending close and abandon the subprocess reaping it was waiting
        for. So a timed-out close queues no sentinel and the thread is left to finish.
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
        self.assertTrue(w._query_q.empty(), "queued a sentinel that would kill the close")
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
