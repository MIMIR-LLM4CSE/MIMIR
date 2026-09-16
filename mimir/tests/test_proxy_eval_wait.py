"""``proxy_eval(op='run')`` waits for the run instead of returning a handle.

The old shape spent one turn launching and then a turn per poll to learn "still
running" — the interesting moment of an optimization run is its end, and the
ratchet verdict is the thing worth reading. These tests pin the three exits:
the run finishes inside the budget, the budget runs out, and the caller asked
to detach up front.

Run:
    python -m unittest mimir.tests.test_proxy_eval_wait -v
"""

import asyncio
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from mimir.tests.test_proxy_ops import eval_session, server_proxy

_LAUNCHED = {
    "status":         "ok",
    "run_dir":        "/tmp/session/run-1",
    "pid":            4242,
    "log":            "/tmp/session/run-1/log",
    "proxy_name":     "tiny",
    "benchmark_name": "bench",
    "next_step":      "ignored",
}
_RESULTS = {"status": "ok", "verdict": "accept", "recommendation": "Accepted",
            "next_step": "edit and run again"}


class _StateSequence:
    """``_run_state`` stand-in yielding a scripted sequence, holding the last value."""

    def __init__(self, states):
        self._states = list(states)
        self.calls = 0

    def __call__(self, run_dir):
        self.calls += 1
        state = self._states[min(self.calls - 1, len(self._states) - 1)]
        return {"state": state, "pid": 1, "slurm_job_id": None, "elapsed_s": 1.0}


class RunAwaitedTests(unittest.TestCase):
    def test_waits_then_returns_the_verdict_inline(self) -> None:
        states = _StateSequence(["running", "running", "done"])
        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", states), \
             patch.object(eval_session, "results", return_value=dict(_RESULTS)), \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            res = asyncio.run(eval_session.run_awaited("tiny"))

        # It kept looking until the run was over, rather than answering on the first read.
        self.assertEqual(states.calls, 3)
        self.assertEqual(res["verdict"], "accept")
        # No polling call is proposed: the answer is already here.
        self.assertNotIn("proxy_eval_status", res["next_step"])
        self.assertNotIn("background_job", res)
        # The launch identity survives onto the verdict — a crash needs the log path.
        self.assertEqual(res["run_dir"], _LAUNCHED["run_dir"])
        self.assertEqual(res["log"], _LAUNCHED["log"])

    def test_a_crashed_run_ends_the_wait(self) -> None:
        states = _StateSequence(["running", "crashed"])
        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", states), \
             patch.object(eval_session, "results", return_value=dict(_RESULTS)) as results, \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            asyncio.run(eval_session.run_awaited("tiny"))

        self.assertEqual(states.calls, 2)
        results.assert_called_once()

    def test_budget_exhaustion_detaches_rather_than_timing_out(self) -> None:
        """A run that outlives the wait is handed over, never abandoned."""
        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", _StateSequence(["running"])), \
             patch.object(eval_session, "results") as results, \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            res = asyncio.run(eval_session.run_awaited("tiny", wait_s=0.05))

        results.assert_not_called()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["background_job"]["run_dir"], _LAUNCHED["run_dir"])
        self.assertIn("end your turn", res["next_step"])

    def test_background_skips_the_wait_entirely(self) -> None:
        launched = dict(_LAUNCHED)
        launched["background_job"] = {"run_dir": _LAUNCHED["run_dir"]}
        states = _StateSequence(["running"])
        with patch.object(eval_session, "run", return_value=launched) as run, \
             patch.object(eval_session, "_run_state", states):
            res = asyncio.run(eval_session.run_awaited("tiny", background=True))

        self.assertEqual(states.calls, 0)  # never even looked
        run.assert_called_once_with("tiny", background=True)
        self.assertIn("background_job", res)

    def test_a_failed_launch_is_not_waited_on(self) -> None:
        states = _StateSequence(["running"])
        with patch.object(eval_session, "run",
                          return_value={"status": "error", "error": "no session"}), \
             patch.object(eval_session, "_run_state", states):
            res = asyncio.run(eval_session.run_awaited("tiny"))

        self.assertEqual(states.calls, 0)
        self.assertEqual(res["status"], "error")


class DivertedWaitTests(unittest.TestCase):
    """The user can end the wait without ending the run.

    Both exits from the wait reach the same place by design — one detached run, one
    watcher, one resume. What differs is the note, because "you asked for this" and
    "it outlived the budget" are not the same thing to the person reading it.
    """

    def setUp(self) -> None:
        self._state = tempfile.mkdtemp(prefix="mimir-eval-divert-")
        self._orig = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._state

    def tearDown(self) -> None:
        if self._orig is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig
        shutil.rmtree(self._state, ignore_errors=True)

    def _divert(self) -> None:
        """Stand in for the click: name the run in the channel's request file."""
        base = eval_session.run_channel._dir(eval_session._DIVERT_CHANNEL)
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "divert"), "w") as fh:
            fh.write(os.path.basename(_LAUNCHED["run_dir"]))

    def test_a_divert_request_ends_the_wait_and_keeps_the_run(self) -> None:
        states = _StateSequence(["running"])

        def _state_then_divert(run_dir):
            # The click lands while the run is going, which is the only moment it can.
            out = states(run_dir)
            self._divert()
            return out

        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", _state_then_divert), \
             patch.object(eval_session, "results") as results, \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            res = asyncio.run(eval_session.run_awaited("tiny", wait_s=30.0))

        # It stopped waiting long before the budget, and settled nothing: the run is
        # still going, so there is no verdict to read.
        results.assert_not_called()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["background_job"]["run_dir"], _LAUNCHED["run_dir"])
        self.assertIn("moved this run to the background", res["note"])
        self.assertNotIn("Still running after", res["note"])

    def test_the_wait_publishes_the_run_for_the_client_to_find(self) -> None:
        # Nothing can be diverted that was never announced.
        seen = {}

        def _capture(run_dir):
            seen["run"] = eval_session.run_channel._read_current(
                eval_session._DIVERT_CHANNEL)
            return {"state": "done", "pid": 1, "slurm_job_id": None, "elapsed_s": 1.0}

        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", _capture), \
             patch.object(eval_session, "results", return_value=dict(_RESULTS)), \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            asyncio.run(eval_session.run_awaited("tiny"))

        self.assertIsNotNone(seen["run"])
        self.assertEqual(seen["run"]["job_key"], os.path.basename(_LAUNCHED["run_dir"]))
        self.assertEqual(seen["run"]["pid"], _LAUNCHED["pid"])

    def test_the_announcement_is_retracted_when_the_wait_ends(self) -> None:
        # A stale announcement would let a later click detach whatever ran next.
        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", _StateSequence(["done"])), \
             patch.object(eval_session, "results", return_value=dict(_RESULTS)), \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            asyncio.run(eval_session.run_awaited("tiny"))

        self.assertIsNone(eval_session.run_channel._read_current(
            eval_session._DIVERT_CHANNEL))

    def test_the_phase_is_republished_for_the_client(self) -> None:
        phases = iter([
            {"state": "running", "pid": 1, "slurm_job_id": None, "elapsed_s": 1.0,
             "phase": "building tiny", "percent": 42.0},
            {"state": "done", "pid": 1, "slurm_job_id": None, "elapsed_s": 2.0},
        ])
        seen = []

        def _next(run_dir):
            st = next(phases)
            seen.append(eval_session.run_channel._read_current(
                eval_session._DIVERT_CHANNEL))
            return st

        with patch.object(eval_session, "run", return_value=dict(_LAUNCHED)), \
             patch.object(eval_session, "_run_state", _next), \
             patch.object(eval_session, "results", return_value=dict(_RESULTS)), \
             patch.object(eval_session, "_RUN_POLL_S", 0.01):
            asyncio.run(eval_session.run_awaited("tiny"))

        # The second read happens after the first tick published what it learned.
        self.assertEqual(seen[1]["phase"], "building tiny")
        self.assertEqual(seen[1]["percent"], 42.0)


class ToolDispatchTests(unittest.TestCase):
    def test_the_tool_awaits_the_run(self) -> None:
        """``proxy_eval`` is a coroutine function, and op='run' awaits the wait."""
        async def _fake(proxy_name="", background=False):
            return {"status": "ok", "verdict": "accept"}

        with patch.object(eval_session, "run_awaited", _fake):
            res = asyncio.run(server_proxy.proxy_eval(op="run", confirm=True))

        self.assertEqual(res["verdict"], "accept")

    def test_the_declared_budget_outlasts_the_wait(self) -> None:
        """The tool's wall must not cut off a wait the session is still inside."""
        self.assertGreater(server_proxy._EVAL_RUN_TIMEOUT,
                           eval_session._RUN_WAIT_BUDGET_S)


if __name__ == "__main__":
    unittest.main()
