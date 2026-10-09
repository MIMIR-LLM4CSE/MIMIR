"""When a detached server is no longer needed — and the many ways it still is.

The criterion is positive, and that is the whole design. "Not busy" is not "has
finished": a worker is also not busy between queries, after a turn that broke, and while
it waits behind a card. An idle test built on the absence of noise stops a server whose
turn merely paused. So a conversation counts as finished only once it has *concluded* —
delivered a final answer — and a run launched but not collected holds the process open
even with no turn running, which is exactly the "I submitted a two-hour build and left"
case the whole feature exists for.
"""
import asyncio
import json
import os
import queue as _queue
import tempfile
import time
import unittest
from unittest import mock

from mimir.client.ui.ws import job_scan
from mimir.client.ui.ws.event_bus import _EventBus
from mimir.client.ui.ws.ws_pool import _AgentPool


class _Worker:
    """Only what the predicate asks of a worker."""

    def __init__(self, *, pending: bool = False, prompt=None, deferral: bool = False):
        self.out_q = _queue.Queue()
        self.session_id = "s1"
        self._query_session_id = None
        self._pending = pending
        self._pending_prompt = prompt
        self.has_deferral = deferral
        self.unattended_since = None

    def has_work_pending(self) -> bool:
        return self._pending

    def drain(self) -> list:
        out = []
        while not self.out_q.empty():
            ev = self.out_q.get_nowait()
            ev.setdefault("session_id", self.session_id)
            out.append(ev)
        return out


class _IdleCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        for target in (job_scan,):
            patcher = mock.patch.object(target, "_MIMIR_DIR_WS", self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _pool(self, workers: dict | None = None) -> _AgentPool:
        pool = object.__new__(_AgentPool)
        pool.active_session_id = None
        pool._workers = dict(workers or {})
        pool._last_use = {}
        pool._queue = []
        pool._building = {}
        pool.stop_requested = None
        pool.server_idle_ttl = 100.0
        pool._idle_since = None
        pool._wakes_pending = {}
        pool._held_checkin = {}
        pool.bus = _EventBus(pool)
        return pool

    def _conclude(self, pool: _AgentPool, session_id: str, worker: _Worker) -> None:
        """Let a turn of *session_id* actually finish, the way the pump sees it."""
        worker.out_q.put({"type": "answer", "text": "done", "session_id": session_id})
        pool.bus.pump_once()

    def _live_job(self, session_id: str, job_key: str) -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": job_key, "command": "make -j8", "pid": os.getpid(),
                       "pid_starttime": job_scan._proc_starttime(os.getpid())}, fh)

    def _ended_job(self, session_id: str, job_key: str) -> None:
        self._live_job(session_id, job_key)
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        with open(os.path.join(job_dir, "exit_code"), "w") as fh:
            fh.write("0")

    def _slurm_job(self, session_id: str, job_id: str, state: str | None = None,
                   dir_name: str = "20261008T190000Z-ab12") -> str:
        """A Slurm submission as ``sbatch_submit`` leaves it: a directory and an id."""
        job_dir = os.path.join(self._tmp.name, "sessions", session_id,
                               "hpc_jobs", dir_name)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "slurm_job_id"), "w") as fh:
            fh.write(job_id)
        if state is not None:
            with open(os.path.join(job_dir, "slurm_state"), "w") as fh:
                fh.write(state)
        return job_dir

    def _baseline(self, session_id: str) -> None:
        """Say this session has been scanned before — the ordinary state."""
        path = os.path.join(self._tmp.name, "sessions", session_id)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, ".wake_baseline"), "w") as fh:
            fh.write("0")

    def _deliver(self, session_id: str, job_key: str) -> None:
        """What a consumer does once the turn carrying a wake is submitted."""
        job_scan.mark_wakes_reported(
            [{"session_id": session_id, "job_key": job_key, "server": "bash"}])


class WhatCountsAsFinishedTests(_IdleCase):
    def test_a_concluded_conversation_with_no_jobs_is_idle(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self.assertTrue(pool.idle_report()["idle"], pool.idle_report()["reasons"])

    def test_a_conversation_that_never_answered_is_not_idle(self):
        # The case "not busy" gets wrong: a turn that ended without saying so. Nothing
        # observed it finishing, so nothing may conclude that it did.
        pool = self._pool({"s1": _Worker()})
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("never delivered an answer" in r for r in report["reasons"]))

    def test_an_error_is_an_ending_too(self):
        # Conflating "concluded" with "succeeded" would leave a failed session looking
        # busy for ever.
        w = _Worker()
        pool = self._pool({"s1": w})
        w.out_q.put({"type": "error", "text": "backend gone", "session_id": "s1"})
        pool.bus.pump_once()
        self.assertTrue(pool.idle_report()["idle"])

    def test_a_turn_in_flight_is_not_idle(self):
        w = _Worker(pending=True)
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)   # an earlier turn did finish
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("in flight" in r for r in report["reasons"]))

    def test_producing_again_un_concludes_the_conversation(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self.assertTrue(pool.idle_report()["idle"])
        w.out_q.put({"type": "output", "text": "a new turn speaks\n"})
        pool.bus.pump_once()
        self.assertFalse(pool.idle_report()["idle"])


class SlurmSubmissionsEndTests(_IdleCase):
    """A Slurm job holds the process open, and stops holding it.

    The second half is the one that was missing, and it cost a night: a submission
    records an id and an id never stops existing, so a workspace that had ever submitted
    one job had a server that could never be idle — whatever it was or was not doing,
    however long ago the job ended. The process does not decide a Slurm job is over by
    inspection; it reads the state a poll wrote down.
    """

    def test_a_submitted_job_holds_the_process_open(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._slurm_job("s1", "123456")
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("still going" in r for r in report["reasons"]))

    def test_a_job_slurm_has_finished_does_not(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._baseline("s1")
        self._slurm_job("s1", "123456", state="done")
        job_scan.mark_job_reported("s1", "123456", "hpc")
        self.assertTrue(pool.idle_report()["idle"], pool.idle_report()["reasons"])

    def test_a_finished_job_still_owed_its_wake_keeps_holding(self):
        # Settled is not delivered: the conversation is still owed the turn, which is
        # the one thing that must outlast the job itself.
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._baseline("s1")
        self._slurm_job("s1", "123456", state="done")
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("nobody has taken in" in r for r in report["reasons"]),
                        report["reasons"])

    def test_a_submission_nothing_can_settle_stops_holding_eventually(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._baseline("s1")
        job_dir = self._slurm_job("s1", "123456")
        aged = time.time() - 30 * 24 * 3600
        os.utime(os.path.join(job_dir, "slurm_job_id"), (aged, aged))
        job_scan.mark_job_reported("s1", "123456", "hpc")
        self.assertTrue(pool.idle_report()["idle"], pool.idle_report()["reasons"])


class JobsHoldItOpenTests(_IdleCase):
    def test_a_live_run_keeps_the_process_needed(self):
        # The case the whole feature exists for: no turn running, and a two-hour build
        # still going.
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._live_job("s1", "j1")
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("still going" in r for r in report["reasons"]))

    def test_a_finished_run_still_owed_a_wake_keeps_the_process_needed(self):
        # The debt one step on from a live run: the job ended, and the turn its result
        # was supposed to start has not happened. Stopping here is how a conversation
        # loses the night — the server shuts down owing a turn it never began.
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._ended_job("s1", "j1")
        self._baseline("s1")
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("nobody has taken in" in r for r in report["reasons"]),
                        report["reasons"])

    def test_a_finished_run_whose_wake_was_delivered_does_not(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._ended_job("s1", "j1")
        self._baseline("s1")
        self._deliver("s1", "j1")
        self.assertTrue(pool.idle_report()["idle"], pool.idle_report()["reasons"])

    def test_a_finished_run_of_a_never_scanned_session_does_not(self):
        # A job directory is never swept, however old, so an unbaselined session holds
        # every run it ever finished with no marker on any of them. Reading that history
        # as debt would leave the workspace permanently un-stoppable — worse than
        # stopping one with a wake outstanding, and the trade the baseline exists for.
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        self._ended_job("s1", "j1")
        self.assertTrue(pool.idle_report()["idle"], pool.idle_report()["reasons"])

    def test_a_run_of_a_conversation_with_no_agent_still_counts(self):
        # A run outlives the agent that launched it, and its session may have no worker
        # at all right now — which is why this is read from disk and not from the pool.
        pool = self._pool()
        self._live_job("gone-session", "j1")
        self.assertFalse(pool.idle_report()["idle"])

    def test_an_unreadable_state_dir_is_treated_as_busy(self):
        # Unknown must not read as "nothing is running": erring the other way stops a
        # server mid-build.
        pool = self._pool()
        # The name bound in ws_pool, not the one in job_scan: the import is by value,
        # so patching the source module would leave the caller untouched and the test
        # would pass for the wrong reason.
        with mock.patch("mimir.client.ui.ws.ws_pool.scan_all_sessions",
                        side_effect=OSError("gone")):
            report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("unreadable" in r for r in report["reasons"]))


class ProhibitionsTests(_IdleCase):
    def test_an_attached_client_forbids_it(self):
        async def run():
            w = _Worker()
            pool = self._pool({"s1": w})
            self._conclude(pool, "s1", w)
            pool.bus.subscribe()
            report = pool.idle_report()
            self.assertFalse(report["idle"])
            self.assertTrue(any("attached" in r for r in report["reasons"]))
        asyncio.run(run())

    def test_a_parked_card_forbids_it(self):
        w = _Worker(prompt={"id": "a1"})
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("parked" in r for r in report["reasons"]))

    def test_an_owed_answer_forbids_it(self):
        w = _Worker(deferral=True)
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("owed" in r for r in report["reasons"]))

    def test_a_turn_waiting_for_a_slot_forbids_it(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        pool._queue = [("s2", lambda _w: None)]
        self.assertFalse(pool.idle_report()["idle"])

    def test_an_agent_being_built_forbids_it(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        self._conclude(pool, "s1", w)
        pool._building = {"s2": object()}
        self.assertFalse(pool.idle_report()["idle"])

    def test_a_worker_that_cannot_be_asked_forbids_it(self):
        class _Broken(_Worker):
            def has_work_pending(self):
                raise RuntimeError("thread is wedged")

        pool = self._pool({"s1": _Broken()})
        self.assertFalse(pool.idle_report()["idle"])

    def test_an_empty_pool_with_nothing_on_disk_is_idle(self):
        self.assertTrue(self._pool().idle_report()["idle"])


class TheClockTests(_IdleCase):
    def _idle_pool(self):
        w = _Worker()
        pool = self._pool({"s1": w})
        pool.stop_requested = asyncio.Event()
        self._conclude(pool, "s1", w)
        return pool, w

    def test_the_clock_starts_rather_than_stopping_immediately(self):
        async def run():
            pool, _w = self._idle_pool()
            pool._consider_stopping()
            self.assertIsNotNone(pool._idle_since)
            self.assertFalse(pool.stop_requested.is_set())
        asyncio.run(run())

    def test_it_stops_once_the_ttl_has_passed(self):
        async def run():
            pool, _w = self._idle_pool()
            pool.server_idle_ttl = 0.0
            pool._consider_stopping()      # starts the clock
            pool._consider_stopping()      # and now the TTL has passed
            self.assertTrue(pool.stop_requested.is_set())
        asyncio.run(run())

    def test_activity_resets_the_clock_rather_than_shortening_it(self):
        # "Idle throughout", not "idle at some point": the difference between stopping
        # a forgotten server and stopping one between two turns.
        async def run():
            pool, w = self._idle_pool()
            pool._consider_stopping()
            started = pool._idle_since
            self.assertIsNotNone(started)
            w.out_q.put({"type": "output", "text": "back to work\n"})
            pool.bus.pump_once()
            pool._consider_stopping()
            self.assertIsNone(pool._idle_since)
            self._conclude(pool, "s1", w)
            time.sleep(0.01)
            pool._consider_stopping()
            self.assertIsNotNone(pool._idle_since)
            self.assertNotEqual(pool._idle_since, started)
            self.assertFalse(pool.stop_requested.is_set())
        asyncio.run(run())

    def test_nothing_stops_when_no_stop_was_wired(self):
        # A pool driven synchronously by a test has no serve loop to ask.
        pool, _w = self._idle_pool()
        pool.stop_requested = None
        pool._consider_stopping()          # must not raise

    def test_request_stop_is_idempotent(self):
        async def run():
            pool, _w = self._idle_pool()
            pool.request_stop()
            pool.request_stop()
            self.assertTrue(pool.stop_requested.is_set())
        asyncio.run(run())

    def test_the_ttl_is_configurable_and_floored(self):
        from mimir.client.ui.ws.ws_pool import _server_idle_ttl
        with mock.patch.dict(os.environ, {"MIMIR_SERVER_IDLE_TTL": "300"}):
            self.assertEqual(_server_idle_ttl(), 300.0)
        with mock.patch.dict(os.environ, {"MIMIR_SERVER_IDLE_TTL": "1"}):
            self.assertEqual(_server_idle_ttl(), 60.0, "a floor keeps a tick from "
                                                       "stopping a working server")
        with mock.patch.dict(os.environ, {"MIMIR_SERVER_IDLE_TTL": "soon"}):
            self.assertEqual(_server_idle_ttl(), 7200.0)


class NobodyAskedToKeepItTests(unittest.TestCase):
    """A server exists claimed, or owned by a window. There is no third state.

    What used to be possible was a server in neither: one a window had merely attached
    to, nobody's to kill, still writing to a pipe whose reader had gone. It kept
    working, unobserved, and nobody had asked it to — the fragile state the whole
    detach path exists to avoid.
    """

    def _pool(self, *, claims: dict, unattended: float | None, ever: bool = True):
        from unittest import mock
        from mimir.client.ui.ws import server_registry, ws_pool
        from mimir.client.ui.ws.ws_pool import _AgentPool

        pool = object.__new__(_AgentPool)
        pool.stop_requested = None

        class _Bus:
            ever_attached = ever

            @staticmethod
            def unattended_for():
                return unattended

        pool.bus = _Bus()
        patcher = mock.patch.object(server_registry, "claims", lambda: claims)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(ws_pool, "_detach_grace", lambda: 30.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        return pool

    def test_a_window_still_here_keeps_it_whatever_the_claims(self):
        pool = self._pool(claims={}, unattended=None)
        self.assertFalse(pool._nobody_asked_to_keep_it())

    def test_a_blink_is_not_a_departure(self):
        # Reloading a VS Code window closes and reopens the socket.
        pool = self._pool(claims={}, unattended=5.0)
        self.assertFalse(pool._nobody_asked_to_keep_it())

    def test_gone_past_the_grace_with_no_claim_is_the_end_of_it(self):
        pool = self._pool(claims={}, unattended=45.0)
        self.assertTrue(pool._nobody_asked_to_keep_it())

    def test_one_conversation_asking_is_enough(self):
        # One process serves the workspace, so any claim keeps all of it alive.
        pool = self._pool(claims={"s2": "auto_all"}, unattended=9999.0)
        self.assertFalse(pool._nobody_asked_to_keep_it())

    def test_a_server_nobody_has_connected_to_yet_has_lost_nothing(self):
        # "Its last client left" presupposes one. A server started by hand must not
        # stop itself out from under the window about to attach.
        pool = self._pool(claims={}, unattended=9999.0, ever=False)
        self.assertFalse(pool._nobody_asked_to_keep_it())

    def test_claims_that_cannot_be_read_are_not_read_as_nobody(self):
        from unittest import mock
        from mimir.client.ui.ws import server_registry

        pool = self._pool(claims={}, unattended=9999.0)
        with mock.patch.object(server_registry, "claims",
                               side_effect=OSError("unreadable")):
            self.assertFalse(pool._nobody_asked_to_keep_it())


if __name__ == "__main__":
    unittest.main()
