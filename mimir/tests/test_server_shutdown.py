"""Stopping the server closes the agents' MCP servers, which run in their own process
groups and would otherwise outlive it; and a stopping server is not found by a window
opened meanwhile, nor does it stop the server that window starts. A stub pool.
"""
from __future__ import annotations

import asyncio
import os
import signal
import types
import unittest

from unittest import mock

from mimir.client.ui.ws.server import serving
from mimir.client.ui.ws.server.serving import _run_until_signalled


class _StubPool:
    def __init__(self) -> None:
        self.closed = False
        self.cap = 3
        self.lifecycle = types.SimpleNamespace(stop_requested=None)
        self.events: list[str] = []

    async def aclose_all(self) -> None:
        self.events.append("agents closed")
        self.closed = True


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def _serve_until(self, sig: int, pool: _StubPool, retire=lambda: None) -> None:
        task = asyncio.get_running_loop().create_task(_run_until_signalled(pool, retire))
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

    async def test_the_address_is_withdrawn_before_the_agents_close(self) -> None:
        """Closing the agents takes seconds; a window reopened meanwhile must not find a
        server that is going away, attach to it, and crash on its first message."""
        pool = _StubPool()
        await self._serve_until(signal.SIGTERM, pool,
                                retire=lambda: pool.events.append("address withdrawn"))
        self.assertEqual(pool.events, ["address withdrawn", "agents closed"])

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


class LockWaitTests(unittest.IsolatedAsyncioTestCase):
    """A server started while the previous one is still stopping waits for its lock."""

    def _registry(self, acquire_after: int, live_after: int | None = None):
        calls = {"n": 0}

        def acquire():
            calls["n"] += 1
            return calls["n"] > acquire_after

        def current():
            return {"pid": 1, "url": "ws://x"} if live_after is not None and calls["n"] > live_after else None

        return mock.patch.multiple(serving.server_registry, acquire=acquire, current=current)

    async def test_it_takes_the_lock_once_the_stopping_server_lets_go(self) -> None:
        with self._registry(acquire_after=2), mock.patch.object(serving.asyncio, "sleep", _no_wait):
            self.assertTrue(await serving._acquire_lock())

    async def test_it_attaches_instead_when_a_live_server_answers(self) -> None:
        with self._registry(acquire_after=99, live_after=1), \
                mock.patch.object(serving.asyncio, "sleep", _no_wait):
            self.assertFalse(await serving._acquire_lock())

    async def test_it_gives_up_when_the_lock_is_never_released(self) -> None:
        with self._registry(acquire_after=10**9), mock.patch.object(serving, "_LOCK_WAIT_SECS", 0.0):
            self.assertFalse(await serving._acquire_lock())


async def _no_wait(_secs: float) -> None:
    return None


if __name__ == "__main__":
    unittest.main()
