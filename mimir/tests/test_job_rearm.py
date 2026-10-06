"""A run that finished while nothing was listening still wakes its conversation.

Background runs already survive everything — their own process session, a trap that
records the exit code, a descriptor on disk. What does not survive is the promise to
report them: a watcher is a task on the worker's loop, and a restart cancels it. The
mechanism to re-make that promise existed but needed a turn in which somebody asked
"where is the job at?". These tests pin the half that removes the asking.
"""
import json
import os
import queue as _queue
import tempfile
import time
import unittest
from unittest import mock

from mimir.client.ui.ws import job_scan
from mimir.client.ui.ws.job_scan import DetachedJob, scan_all_sessions, scan_session
from mimir.client.ui.ws.ws_worker import _AgentWorker


class _ScanCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(job_scan, "_MIMIR_DIR_WS", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _baseline(self, session_id: str) -> None:
        """Say this session has been scanned before — the ordinary state.

        Only the very first scan of a session establishes a baseline instead of
        reporting; every test about *reporting* therefore has to be past that point,
        and the first-scan behaviour has its own tests in BaselineTests.
        """
        # Beside the session's other sidecars, not inside ``jobs/``: that directory is
        # read as a list of job handles, and a stray file there reads as a live run.
        path = os.path.join(self._tmp.name, "sessions", session_id)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, ".wake_baseline"), "w") as fh:
            fh.write("0")

    def _job(self, session_id: str, job_key: str, *, pid: int | None = None,
             starttime: int | None = None, exit_code: int | None = None,
             ephemeral: bool = False, command: str = "make -j8",
             meta: dict | None = None) -> str:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir, exist_ok=True)
        if meta is None:
            meta = {"job_key": job_key, "command": command,
                    "pid": pid if pid is not None else os.getpid(),
                    "pid_starttime": starttime, "started_at": time.time(),
                    "ephemeral": ephemeral}
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        if exit_code is not None:
            with open(os.path.join(job_dir, "exit_code"), "w") as fh:
                fh.write(str(exit_code))
        self._baseline(session_id)
        return job_dir

    def _slurm_job(self, session_id: str, dir_name: str, job_id: str) -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id,
                               "hpc_jobs", dir_name)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "slurm_job_id"), "w") as fh:
            fh.write(job_id)

    @staticmethod
    def _dead_pid() -> int:
        """A pid that is certainly not running."""
        # Walk up from an unlikely number until one is free; 0 and negatives are
        # rejected by the liveness check itself and would not exercise it.
        for pid in range(4194300, 4194200, -1):
            try:
                os.kill(pid, 0)
            except OSError:
                return pid
        raise unittest.SkipTest("no free pid to use as a dead one")


class ScanTests(_ScanCase):
    def test_a_running_job_is_reported_live(self):
        # Our own pid, with its real start time: alive, and the start time matches.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()))
        jobs = scan_session("s1")
        self.assertEqual([(j.job_key, j.live, j.state) for j in jobs],
                         [("j1", True, "running")])

    def test_a_recorded_exit_code_wins_over_whatever_the_pid_looks_like(self):
        # The trap writes it as the shell leaves, so a run that got there is finished.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()), exit_code=0)
        job = scan_session("s1")[0]
        self.assertFalse(job.live)
        self.assertEqual(job.state, "done")

    def test_a_non_zero_exit_code_is_a_crash(self):
        self._job("s1", "j1", exit_code=2)
        self.assertEqual(scan_session("s1")[0].state, "crashed")

    def test_a_dead_pid_with_no_exit_code_is_unknown_not_done(self):
        # Claiming an outcome nobody observed is worse than saying it is not knowable.
        self._job("s1", "j1", pid=self._dead_pid())
        job = scan_session("s1")[0]
        self.assertFalse(job.live)
        self.assertEqual(job.state, "unknown")

    def test_a_recycled_pid_is_not_mistaken_for_the_job(self):
        # Our pid is alive, but it started at a different time: a different process
        # wears that number now, and the job is gone.
        self._job("s1", "j1", pid=os.getpid(), starttime=1)
        self.assertFalse(scan_session("s1")[0].live)

    def test_an_ephemeral_job_is_not_a_handle_anyone_was_handed(self):
        # A blocking run's scratch buffer: nothing is waiting to hear about it.
        self._job("s1", "j1", exit_code=0, ephemeral=True)
        self.assertEqual(scan_session("s1"), [])

    def test_an_unreadable_descriptor_is_skipped_not_fatal(self):
        job_dir = os.path.join(self._tmp.name, "sessions", "s1", "jobs", "broken")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            fh.write("{not json")
        self._job("s1", "j2", exit_code=0)
        self.assertEqual([j.job_key for j in scan_session("s1")], ["j2"])

    def test_a_session_with_no_jobs_scans_empty(self):
        self.assertEqual(scan_session("never-ran-anything"), [])
        self.assertEqual(scan_session(""), [])

    def test_a_slurm_job_is_live_until_slurm_says_otherwise(self):
        # Its state is in the controller, not in a pid here, so the only honest reading
        # is "there is a job, ask Slurm" — the watcher's first poll settles it.
        self._slurm_job("s1", "run-1", "12345")
        job = [j for j in scan_session("s1") if j.kind == "slurm"][0]
        self.assertTrue(job.live)
        self.assertEqual(job.status_op()["tool"], "slurm_status")
        self.assertEqual(job.status_op()["args"]["job_id"], "12345")

    def test_a_slurm_dir_with_no_id_is_skipped(self):
        os.makedirs(os.path.join(self._tmp.name, "sessions", "s1", "hpc_jobs", "x"))
        self.assertEqual(scan_session("s1"), [])

    def test_every_session_is_scanned(self):
        self._job("s1", "j1", exit_code=0)
        self._job("s2", "j2", exit_code=1)
        found = scan_all_sessions()
        self.assertEqual(sorted(found), ["s1", "s2"])
        self.assertEqual(found["s2"][0].state, "crashed")

    def test_the_descriptor_names_an_op_to_poll(self):
        # The whole trust boundary for re-registration: a tool to call and a job to
        # call it for.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()))
        d = scan_session("s1")[0].descriptor()
        self.assertEqual(d["status_op"], {"tool": "bash_job",
                                          "args": {"job_key": "j1"}})
        self.assertEqual(d["job_key"], "j1")


class _RearmWorker(_AgentWorker):
    """A worker with only what the re-arm path touches."""

    def __init__(self, session_id: str) -> None:   # noqa: D107 - no agent, no loop
        self.out_q = _queue.Queue()
        self.session_id = session_id
        self.active_session_id = None
        self._query_session_id = None
        self._bg_jobs = {}
        self.registered: list[tuple[dict, str | None]] = []

    def _register_bg_job(self, descriptor, owner=None) -> bool:
        self.registered.append((descriptor, owner))
        return True


class RearmTests(_ScanCase):
    def _worker(self, session_id: str = "s1") -> _RearmWorker:
        return _RearmWorker(session_id)

    def _events(self, worker) -> list[dict]:
        out = []
        while not worker.out_q.empty():
            out.append(worker.out_q.get_nowait())
        return out

    def test_a_live_run_gets_its_watcher_back(self):
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()))
        w = self._worker()
        result = w.rearm_detached_jobs()
        self.assertEqual(result["rearmed"], ["j1"])
        self.assertEqual(self._events(w), [])
        descriptor, owner = w.registered[0]
        self.assertEqual(owner, "s1", "the wake must go to the session that launched it")
        self.assertEqual(descriptor["status_op"]["args"]["job_key"], "j1")

    def test_a_run_that_ended_unwatched_is_reported_as_a_wake(self):
        self._job("s1", "j1", exit_code=0, command="make -j8")
        w = self._worker()
        result = w.rearm_detached_jobs()
        self.assertEqual(result["reported"], ["j1"])
        ev = self._events(w)[0]
        self.assertEqual(ev["type"], "job_complete")
        self.assertEqual(ev["state"], "done")
        self.assertEqual(ev["session_id"], "s1")
        self.assertEqual(ev["summary"]["command"], "make -j8")

    def test_a_crash_is_reported_as_a_crash(self):
        self._job("s1", "j1", exit_code=3)
        w = self._worker()
        w.rearm_detached_jobs()
        self.assertEqual(self._events(w)[0]["state"], "crashed")

    def test_a_run_that_stopped_being_trackable_says_so(self):
        self._job("s1", "j1", pid=self._dead_pid())
        w = self._worker()
        w.rearm_detached_jobs()
        self.assertEqual(self._events(w)[0]["state"], "unknown")

    def test_a_run_already_watched_is_left_alone(self):
        # The ordinary case of a server that simply stayed up: re-arming anything would
        # put a second watcher on a run and wake the conversation twice.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()))
        w = self._worker()
        w._bg_jobs["j1"] = mock.Mock(task=mock.Mock(done=lambda: False))
        result = w.rearm_detached_jobs()
        self.assertEqual(result, {"rearmed": [], "reported": []})
        self.assertEqual(w.registered, [])

    def test_a_watcher_that_has_finished_does_not_count_as_holding_the_run(self):
        self._job("s1", "j1", exit_code=0)
        w = self._worker()
        w._bg_jobs["j1"] = mock.Mock(task=mock.Mock(done=lambda: True))
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["j1"])

    def test_several_runs_are_sorted_into_the_two_piles(self):
        self._job("s1", "live", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()))
        self._job("s1", "ended", exit_code=0)
        w = self._worker()
        result = w.rearm_detached_jobs()
        self.assertEqual(result["rearmed"], ["live"])
        self.assertEqual(result["reported"], ["ended"])

    def test_only_this_sessions_runs_are_touched(self):
        self._job("s1", "mine", exit_code=0)
        self._job("s2", "theirs", exit_code=0)
        w = self._worker("s1")
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["mine"])

    def test_a_worker_with_no_session_does_nothing(self):
        self._job("s1", "j1", exit_code=0)
        w = self._worker("")
        self.assertEqual(w.rearm_detached_jobs(), {"rearmed": [], "reported": []})


class ReportedOnceTests(_ScanCase):
    """Two places look for runs that ended unwatched; one wake must come of it.

    A connection arriving and a worker being built both ask. A run announced twice is a
    conversation woken twice for one build, so the marker lives in the run's own
    directory and outlives whichever process wrote it.
    """

    def test_a_reported_run_is_left_out_of_the_next_scan(self):
        self._job("s1", "j1", exit_code=0)
        job = scan_session("s1")[0]
        self.assertFalse(job_scan.was_reported(job))
        job_scan.mark_reported(job)
        self.assertTrue(job_scan.was_reported(job))
        self.assertEqual(scan_session("s1"), [])

    def test_the_marker_can_be_asked_for_anyway(self):
        self._job("s1", "j1", exit_code=0)
        job_scan.mark_reported(scan_session("s1")[0])
        self.assertEqual([j.job_key for j in
                          scan_session("s1", include_reported=True)], ["j1"])

    def test_a_live_run_is_never_filtered_by_the_marker(self):
        # Re-arming a run already held costs nothing — the worker dedups — whereas
        # filtering one out would leave it unwatched for good.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=job_scan._proc_starttime(os.getpid()))
        job_scan.mark_reported(scan_session("s1")[0])
        self.assertEqual([j.job_key for j in scan_session("s1")], ["j1"])

    def test_the_worker_marks_what_it_reports(self):
        self._job("s1", "j1", exit_code=0)
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["j1"])
        # A second worker for the same session — a rebuild — says nothing again.
        self.assertEqual(_RearmWorker("s1").rearm_detached_jobs()["reported"], [])

    def test_a_marker_that_cannot_be_written_still_lets_the_run_be_announced(self):
        # Costing a duplicate wake is a nuisance; refusing to announce the run because
        # the marker failed is the bug this path exists to fix.
        self._job("s1", "j1", exit_code=0)
        w = _RearmWorker("s1")
        # Only the marker's write: patching `open` would break the scan that finds
        # the run in the first place, and then the test would pass for the wrong reason.
        with mock.patch("mimir.client.ui.ws.job_scan.os.makedirs",
                        side_effect=OSError("read-only")):
            reported = w.rearm_detached_jobs()["reported"]
        self.assertEqual(reported, ["j1"])


class BaselineTests(_ScanCase):
    """The first look at a session establishes what is already known, not a backlog.

    A detached job's directory is never swept, however old, and markers only started
    being written when this did — so an existing workspace has every build it ever ran
    sitting there unmarked. Read as wakes owed, that is a conversation woken for a
    two-month-old build, and unlike a missed wake it is unbounded.
    """

    def _unbaselined_job(self, session_id: str, job_key: str, exit_code: int) -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": job_key, "command": "make", "pid": 1,
                       "pid_starttime": 1}, fh)
        with open(os.path.join(job_dir, "exit_code"), "w") as fh:
            fh.write(str(exit_code))

    def test_the_first_scan_reports_nothing_and_records_that_it_looked(self):
        for i in range(5):
            self._unbaselined_job("s1", f"old-{i}", 0)
        self.assertFalse(job_scan.has_baseline("s1"))
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs(), {"rearmed": [], "reported": []})
        self.assertTrue(job_scan.has_baseline("s1"))
        self.assertTrue(w.out_q.empty(), "woke a conversation for a historical build")

    def test_a_job_that_ends_after_the_baseline_is_reported(self):
        # The sequence that matters: looked at once, then a run finishes while away.
        self._unbaselined_job("s1", "old", 0)
        _RearmWorker("s1").rearm_detached_jobs()          # establishes the baseline
        self._unbaselined_job("s1", "new", 0)
        self.assertEqual(_RearmWorker("s1").rearm_detached_jobs()["reported"], ["new"])

    def test_a_live_run_is_still_re_armed_on_the_first_scan(self):
        # The baseline is about what has *ended*; a run still going needs its watcher
        # back whether or not this session has been seen before.
        job_dir = os.path.join(self._tmp.name, "sessions", "s1", "jobs", "live")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": "live", "command": "sleep 9999",
                       "pid": os.getpid(),
                       "pid_starttime": job_scan._proc_starttime(os.getpid())}, fh)
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs()["rearmed"], ["live"])
        self.assertTrue(job_scan.has_baseline("s1"))

    def test_the_baseline_marks_the_history_so_it_is_never_claimed_later(self):
        self._unbaselined_job("s1", "old", 0)
        _RearmWorker("s1").rearm_detached_jobs()
        jobs = scan_session("s1", include_reported=True)
        self.assertTrue(all(job_scan.was_reported(j) for j in jobs))

    def test_a_watcher_reporting_a_run_marks_it_so_a_restart_does_not_repeat_it(self):
        # The common case and the worse one: a job whose own watcher reported it
        # normally must not be woken again on the next attach.
        self._baseline("s1")
        self._unbaselined_job("s1", "j1", 0)
        job_scan.mark_job_reported("s1", "j1")
        self.assertEqual(_RearmWorker("s1").rearm_detached_jobs()["reported"], [])

    def test_marking_by_key_needs_neither_a_session_nor_a_job(self):
        job_scan.mark_job_reported(None, "j1")
        job_scan.mark_job_reported("s1", "")


class StateWordingTests(unittest.TestCase):
    """``unknown`` is not a tidier word for failure."""

    def test_the_three_terminal_states(self):
        base = dict(session_id="s", job_key="j", kind="shell", live=False)
        self.assertEqual(DetachedJob(**base, exit_code=0).state, "done")
        self.assertEqual(DetachedJob(**base, exit_code=1).state, "crashed")
        self.assertEqual(DetachedJob(**base, exit_code=None).state, "unknown")

    def test_a_live_run_has_no_terminal_state(self):
        self.assertEqual(
            DetachedJob(session_id="s", job_key="j", kind="shell", live=True,
                        exit_code=None).state,
            "running")


if __name__ == "__main__":
    unittest.main()
