"""An unanswered approval must keep the agent parked — never auto-proceed.

Regression test for the WS worker's approval/continue/question shims: they used
to time out after 300 s and return "denied", which let the agent resume on its
own when the user simply hadn't answered yet. The wait is now indefinite but
still cancellable via the agent's cancel flag (the Stop button).
"""

import queue as _queue
import threading
import time
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker


class _StubAgent:
    def __init__(self) -> None:
        self._cancel_flag = threading.Event()


def _make_worker() -> _AgentWorker:
    w = _AgentWorker.__new__(_AgentWorker)  # bypass __init__ (spawns a thread)
    w._agent = _StubAgent()
    return w


class AwaitResponseTests(unittest.TestCase):
    def test_returns_response_when_answered(self) -> None:
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        q.put({"choice": "y"})
        self.assertEqual(w._await_response(q), {"choice": "y"})

    def test_blocks_until_answered_does_not_time_out(self) -> None:
        """No response, no cancel → the call stays blocked (never returns on its own)."""
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        result: list = []

        def _run() -> None:
            result.append(w._await_response(q))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        # Give it well past a poll slice; it must still be waiting (not proceeded).
        t.join(timeout=1.0)
        self.assertTrue(t.is_alive(), "shim should stay parked while unanswered")
        self.assertEqual(result, [])
        # A late answer unblocks it.
        q.put({"choice": "n"})
        t.join(timeout=2.0)
        self.assertEqual(result, [{"choice": "n"}])

    def test_cancel_flag_interrupts_wait(self) -> None:
        """Pressing Stop (cancel flag) unblocks the wait with None (cancelled)."""
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        result: list = []

        def _run() -> None:
            result.append(w._await_response(q))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.05)
        w._agent._cancel_flag.set()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [None])


if __name__ == "__main__":
    unittest.main()
