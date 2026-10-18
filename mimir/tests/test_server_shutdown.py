"""Stopping the server closes the agents' MCP servers, which run in their own process
groups and would otherwise outlive it. A stub pool.
"""
from __future__ import annotations

import asyncio
import os
import signal
import types
import unittest

from mimir.client.ui.ws.server.serving import _run_until_signalled


class _StubPool:
    def __init__(self) -> None:
        self.closed = False
        self.cap = 3
        self.lifecycle = types.SimpleNamespace(stop_requested=None)

    async def aclose_all(self) -> None:
        self.closed = True


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def _serve_until(self, sig: int, pool: _StubPool) -> None:
        task = asyncio.get_running_loop().create_task(_run_until_signalled(pool))
        # Let the handlers install before raising the signal at ourselves.
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), sig)
        await asyncio.wait_for(task, timeout=5)

    async def test_sigterm_closes_every_agent(self) -> None:
        """What the extension sends when it replaces this server."""
        pool = _StubPool()
        await self._serve_until(signal.SIGTERM, pool)
        self.assertTrue(pool.closed, "the MCP servers were left orphaned")

    async def test_sigint_closes_every_agent(self) -> None:
        """Ctrl-C on a server run by hand."""
        pool = _StubPool()
        await self._serve_until(signal.SIGINT, pool)
        self.assertTrue(pool.closed)

    async def test_the_handlers_do_not_outlive_the_server(self) -> None:
        """Left installed they would answer for whatever runs on this loop next."""
        pool = _StubPool()
        await self._serve_until(signal.SIGTERM, pool)
        # Default disposition restored: raising again must not reach a stale handler.
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    async def test_a_loop_without_signal_handlers_still_serves(self) -> None:
        """Windows has no add_signal_handler; the pool's idle release is the reaping there."""
        loop = asyncio.get_running_loop()
        original = loop.add_signal_handler

        def _unsupported(*_a, **_k):
            raise NotImplementedError

        loop.add_signal_handler = _unsupported          # type: ignore[method-assign]
        self.addCleanup(setattr, loop, "add_signal_handler", original)
        pool = _StubPool()
        task = loop.create_task(_run_until_signalled(pool))
        await asyncio.sleep(0.05)
        self.assertFalse(task.done(), "stopped serving with no way to be signalled")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        # Cancelled rather than signalled, and the close still ran on the way out.
        self.assertTrue(pool.closed)


if __name__ == "__main__":
    unittest.main()
