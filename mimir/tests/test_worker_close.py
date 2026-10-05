"""Closing a worker closes its MCP servers, and never under a live turn.

Two failures hide behind one symptom — ``Tool 'bash_run' failed to execute:
ClosedResourceError``. ``stdio_client`` anchors its anyio cancel scope to the task that
entered the context, so exiting the agent's exit stack anywhere else raises and leaves the
stack half-unwound: the streams closed, the subprocess alive. The turn still running on
that agent then fails its next tool call on a dead stream, and nothing was reaped.

So the close must happen in the task that opened the servers, and must not happen at all
while a turn is running or queued. Both are checked here; ``test_agent_pool`` checks what
the pool does with the second answer.

Pure-Python: a stub agent, because what is under test is which task closes it and when.
"""
from __future__ import annotations

import asyncio
import threading
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker


class _StubAgent:
    def __init__(self) -> None:
        self.closed_in: asyncio.Task | None = None
        self.closes = 0

    async def cleanup(self) -> None:
        self.closes += 1
        self.closed_in = asyncio.current_task()


def _bare_worker() -> _AgentWorker:
    """A worker with its fields but no thread — ``detached`` with a session."""
    worker = _AgentWorker.detached("test-model")
    worker.session_id = "s1"
    return worker


class CloseRunsInTheOwningTaskTests(unittest.IsolatedAsyncioTestCase):

    async def test_setup_and_close_share_a_task(self) -> None:
        """The anyio invariant: the task that entered the stack is the one that exits it."""
        worker = _bare_worker()
        agent = _StubAgent()
        opened_in: list[asyncio.Task | None] = []

        async def _setup() -> None:
            opened_in.append(asyncio.current_task())
            worker._agent = agent
            worker._ready.set()

        async def _query_loop() -> None:
            return

        worker._setup = _setup                      # type: ignore[assignment]
        worker._query_loop = _query_loop            # type: ignore[assignment]

        await worker._live()

        self.assertEqual(agent.closes, 1, "the MCP servers were left open")
        self.assertIs(agent.closed_in, opened_in[0],
                      "closed from a task that did not open the servers")

    async def test_a_failed_setup_closes_what_it_had_connected(self) -> None:
        """Half of ~19 servers connected is still half of ~19 subprocesses."""
        worker = _bare_worker()
        agent = _StubAgent()

        async def _setup() -> None:
            # What the real one does: hold the half-built agent, then fail.
            try:
                raise RuntimeError("backend never came up")
            except Exception as exc:
                worker._error = exc
                await agent.cleanup()
            finally:
                worker._ready.set()

        worker._setup = _setup                      # type: ignore[assignment]
        ran = []
        worker._query_loop = lambda: ran.append(1)  # type: ignore[assignment]

        await worker._live()
        self.assertEqual(agent.closes, 1)
        self.assertEqual(ran, [], "served queries on an agent that never came up")

    async def test_the_query_loop_ending_closes_the_agent(self) -> None:
        """``shutdown``'s sentinel is the whole close mechanism."""
        worker = _bare_worker()
        agent = _StubAgent()

        async def _setup() -> None:
            worker._agent = agent
            worker._ready.set()

        async def _query_loop() -> None:
            # The real loop returns on the sentinel; here it is already waiting.
            self.assertIsNone(worker._query_q.get_nowait())

        worker._setup = _setup                      # type: ignore[assignment]
        worker._query_loop = _query_loop            # type: ignore[assignment]
        worker.shutdown()

        await worker._live()
        self.assertEqual(agent.closes, 1)
        self.assertTrue(worker._closed.is_set(), "nothing told the caller it was closed")
        self.assertIsNone(worker._agent, "a closed agent still reads as ready")


class WorkPendingTests(unittest.TestCase):
    """What eviction must read. ``is_busy`` answers a narrower question."""

    def setUp(self) -> None:
        self.worker = _bare_worker()

    def test_an_idle_worker_has_no_work(self) -> None:
        self.assertFalse(self.worker.has_work_pending())

    def test_a_queued_turn_counts(self) -> None:
        """The gap the bug lived in: submitted, not yet picked up, and is_busy says no."""
        self.worker.submit_query("do a thing", [], session_id="s1")
        self.assertFalse(self.worker.is_busy())
        self.assertTrue(self.worker.has_work_pending(),
                        "a turn about to start would be evicted under itself")

    def test_a_running_turn_counts(self) -> None:
        self.worker._current_task = object()        # type: ignore[assignment]
        self.assertTrue(self.worker.has_work_pending())

    def test_the_sentinel_reads_as_work(self) -> None:
        """``shutdown`` queues None; that is the close asking, not a turn."""
        self.worker.shutdown()
        self.assertTrue(self.worker.has_work_pending(),
                        "the sentinel sits on the same queue, so this is conservative")


class AcloseWithoutALoopTests(unittest.TestCase):
    """``aclose`` on a worker that never built an agent must not hang or raise."""

    def test_no_agent_no_wait(self) -> None:
        worker = _bare_worker()
        done = threading.Event()
        threading.Thread(target=lambda: (worker.aclose(timeout=1.0), done.set()),
                         daemon=True).start()
        self.assertTrue(done.wait(5.0), "aclose waited for a close that cannot happen")
        self.assertIsNone(worker._query_q.get_nowait(), "no sentinel was sent")


if __name__ == "__main__":
    unittest.main()
