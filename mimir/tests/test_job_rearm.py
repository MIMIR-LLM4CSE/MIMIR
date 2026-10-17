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

from mimir.servers._shared.processes import proc_starttime
from mimir.client.session import jobs_on_disk
from mimir.client.session.jobs_on_disk import DetachedJob, scan_all_sessions, scan_session
from mimir.client.ui.ws.job_watcher import JobWatcher
from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests.support.state import pin_state_dir


def _hpc_tool_names() -> set[str]:
    """The tools the HPC server actually registers, read off its source.

    Read rather than imported: this is a client-side test, and what it needs to know is
    one fact about the other tree — that the name a re-armed watcher will call is a name
    something answers to. Nothing else here should have to import an MCP server.
    """
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]
              / "servers" / "hpc" / "server_hpc.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(call, ast.Attribute) and call.attr == "tool":
                names.add(node.name)
    return names


class _ScanCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        pin_state_dir(self, self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _events(self, worker) -> list[dict]:
        out = []
        while not worker.out_q.empty():
            out.append(worker.out_q.get_nowait())
        return out

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

    def _slurm_job(self, session_id: str, dir_name: str, job_id: str,
                   state: str | None = None, watcher: dict | None = None) -> str:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id,
                               "hpc_jobs", dir_name)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "slurm_job_id"), "w") as fh:
            fh.write(job_id)
        if state is not None:
            with open(os.path.join(job_dir, "slurm_state"), "w") as fh:
                fh.write(state)
        if watcher is not None:
            with open(os.path.join(job_dir, "watcher.json"), "w") as fh:
                json.dump(watcher, fh)
        self._baseline(session_id)
        return job_dir

    @staticmethod
    def _proxy_watcher(run_name: str = "run-0001") -> dict:
        """The handle a proxy run's own server watches it with, as it records it."""
        return {"server": "proxy", "run_dir": f"/w/proxy_bench/runs/{run_name}",
                "job_key": run_name, "kind": "proxy-eval",
                "status_op": {"tool": "proxy_eval_status",
                              "args": {"proxy_name": "solver"}},
                "summary_op": {"tool": "proxy_eval_report", "args": {}}}

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
                  starttime=proc_starttime(os.getpid()))
        jobs = scan_session("s1")
        self.assertEqual([(j.job_key, j.live, j.state) for j in jobs],
                         [("j1", True, "running")])

    def test_a_recorded_exit_code_wins_over_whatever_the_pid_looks_like(self):
        # The trap writes it as the shell leaves, so a run that got there is finished.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()), exit_code=0)
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
        self.assertEqual(job.status_op()["args"]["job_id"], "12345")

    def test_the_slurm_status_op_names_a_tool_the_server_has(self):
        # The tool name is the whole of the promise: a watcher polling one that does not
        # exist reads five unreadable answers, gives the run up, and settles nothing —
        # so the next scan re-arms it, for as long as the process lives.
        self._slurm_job("s1", "run-1", "12345")
        job = [j for j in scan_session("s1") if j.kind == "slurm"][0]
        self.assertEqual(job.status_op()["tool"], "slurm_job_status")
        self.assertIn(job.status_op()["tool"], _hpc_tool_names())

    def test_a_recorded_terminal_state_ends_a_slurm_job(self):
        # What the exit-code trap is to a shell job. Without it the directory says
        # "there is a job" for ever: an id never stops existing.
        self._slurm_job("s1", "run-1", "12345", state="done")
        job = [j for j in scan_session("s1") if j.kind == "slurm"][0]
        self.assertFalse(job.live)
        self.assertEqual(job.state, "done")

    def test_a_crashed_slurm_job_reports_as_crashed(self):
        self._slurm_job("s1", "run-1", "12345", state="crashed")
        self.assertEqual(scan_session("s1")[0].state, "crashed")

    def test_a_slurm_job_slurm_has_forgotten_is_over_without_an_outcome(self):
        # 'unknown' is terminal and claims nothing: the scheduler let go, and no outcome
        # was ever observed. Claiming 'done' here would invent one.
        self._slurm_job("s1", "run-1", "12345", state="unknown")
        job = scan_session("s1")[0]
        self.assertFalse(job.live)
        self.assertEqual(job.state, "unknown")
        self.assertIsNone(job.exit_code)

    def test_an_unreadable_state_word_is_not_a_settlement(self):
        self._slurm_job("s1", "run-1", "12345", state="PENDING-ish")
        self.assertTrue(scan_session("s1")[0].live)

    def test_an_unsettled_submission_ages_out(self):
        # The case nothing will ever settle: no conversation left to poll the job, or a
        # machine whose Slurm tools have gone. Bounded so one submission cannot hold a
        # workspace open for ever.
        self._slurm_job("s1", "run-1", "12345")
        id_path = os.path.join(self._tmp.name, "sessions", "s1", "hpc_jobs",
                               "run-1", "slurm_job_id")
        aged = time.time() - 2 * 3600
        os.utime(id_path, (aged, aged))
        self.assertTrue(scan_session("s1")[0].live)   # two hours is no evidence at all
        with mock.patch.dict(os.environ, {"MIMIR_SLURM_STALE_AFTER": "3600"}):
            job = scan_session("s1")[0]
        self.assertFalse(job.live)
        self.assertEqual(job.state, "unknown")

    def test_a_marker_lands_in_the_submission_own_directory(self):
        # A Slurm key is an id, not a directory name, so marking has to resolve it back
        # to the descriptor that holds it.
        self._slurm_job("s1", "run-1", "12345", state="done")
        jobs_on_disk.mark_job_reported("s1", "12345", "hpc")
        self.assertTrue(os.path.exists(os.path.join(
            self._tmp.name, "sessions", "s1", "hpc_jobs", "run-1", "reported")))
        self.assertEqual([j for j in scan_session("s1") if j.kind == "slurm"], [])

    def test_a_marker_written_under_the_bare_id_still_counts(self):
        # Where markers landed before the id was resolved. Reading only the resolved
        # directory would announce every one of those runs a second time.
        self._slurm_job("s1", "run-1", "12345", state="done")
        legacy = os.path.join(self._tmp.name, "sessions", "s1", "hpc_jobs", "12345")
        os.makedirs(legacy, exist_ok=True)
        with open(os.path.join(legacy, "reported"), "w") as fh:
            fh.write("1")
        self.assertEqual([j for j in scan_session("s1") if j.kind == "slurm"], [])

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
        # call it for. The directory travels with them because a key alone resolves
        # against the reading session, and a sub-agent's run is not under it.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()))
        job = scan_session("s1")[0]
        d = job.descriptor()
        self.assertEqual(d["status_op"],
                         {"tool": "bash_job",
                          "args": {"job_key": "j1", "run_dir": job.run_dir}})
        self.assertEqual(d["job_key"], "j1")
        self.assertTrue(d["run_dir"].endswith("/s1/jobs/j1"))


class _RearmWorker(_AgentWorker):
    """A worker with only what the re-arm path touches."""

    def __init__(self, session_id: str) -> None:   # noqa: D107 - no agent, no loop
        self.out_q = _queue.Queue()
        self.session_id = session_id
        self.active_session_id = None
        self._query_session_id = None
        self.jobs = JobWatcher(self)
        self.registered: list[tuple[dict, str | None]] = []
        self.jobs.register = self._record

    def _record(self, descriptor, owner=None) -> bool:
        self.registered.append((descriptor, owner))
        return True


class ASubAgentsRunsAreFoundTests(_ScanCase):
    """A child's runs, found under its parent.

    A child's own watcher goes with the child, and one killed outright hands nothing
    over — so this scan is the only thing left that can find what it left going. Its
    jobs are two levels down (``<parent>/subagents/<child>/jobs/<key>``), which a
    single-level listing never reached: the reader could read the nested path all along,
    the enumeration simply never offered it one.
    """

    def test_a_childs_jobs_are_found_under_its_parent(self):
        self._job("s1", "j1", exit_code=0)
        self._job("s1/subagents/sub-1a2b", "j2", exit_code=1)
        found = scan_all_sessions()
        self.assertEqual(sorted(found), ["s1", "s1/subagents/sub-1a2b"])
        self.assertEqual(found["s1/subagents/sub-1a2b"][0].state, "crashed")

    def test_nothing_deeper_than_subagents_is_walked(self):
        """One named level, because that is the whole contract a child writes.

        A child cannot delegate, so there is never a third level — and walking
        arbitrary depth would make the scan's cost a property of someone's directory
        tree rather than of the work.
        """
        self._job("s1/subagents/sub-1a2b/subagents/sub-deep", "j3", exit_code=0)
        self.assertEqual(sorted(scan_all_sessions()), [])

    def test_a_childs_run_is_the_parents_to_hear_about(self):
        """Whose run it is and who gets woken are two different answers.

        Nothing can load a sub-session as a conversation: ``_sub_session_id`` writes
        directories and no ``<id>.json``. The person is watching the parent, and the
        parent's agent is the one that can act on what came back.
        """
        self._job("s1/subagents/sub-1a2b", "j2", exit_code=0)
        job = scan_session("s1/subagents/sub-1a2b")[0]
        self.assertEqual(job.session_id, "s1/subagents/sub-1a2b")
        self.assertEqual(job.owner_session, "s1")

    def test_a_childs_wake_is_settled_where_the_scan_looks(self):
        # Woken in the parent, marked under the child: otherwise every scan re-announces it.
        self._job("s1/subagents/sub-1a2b", "j2", exit_code=0)
        ev = {"job_key": "j2", "server": "bash", "session_id": "s1",
              "run_session": "s1/subagents/sub-1a2b"}
        self.assertFalse(jobs_on_disk.wake_reported(ev))
        jobs_on_disk.mark_wakes_reported([ev])
        self.assertTrue(jobs_on_disk.wake_reported(ev))
        self.assertEqual(scan_session("s1/subagents/sub-1a2b"), [])

    def test_a_run_of_its_own_is_owned_by_the_session_that_holds_it(self):
        self._job("s1", "j1", exit_code=0)
        self.assertEqual(scan_session("s1")[0].owner_session, "s1")

    def test_the_descriptor_names_the_childs_directory(self):
        """So the parent polls the run where it actually is.

        A bare handle resolves against the reading session and comes back "no such
        job", which the job layer spells ``unknown`` — terminal. The parent would
        announce a healthy build as lost on its first poll.
        """
        self._job("s1/subagents/sub-1a2b", "j2", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()))
        d = scan_session("s1/subagents/sub-1a2b")[0].descriptor()
        self.assertTrue(
            d["status_op"]["args"]["run_dir"].endswith("/subagents/sub-1a2b/jobs/j2"))


class AChildsBaselineIsItsOwnTests(_ScanCase):
    """A child seen for the first time brings no backlog of its own."""

    def _job_unbaselined(self, session_id: str, job_key: str, **kw) -> str:
        job_dir = self._job(session_id, job_key, **kw)
        os.remove(jobs_on_disk._baseline_path(session_id))
        return job_dir

    def test_the_first_scan_of_a_child_wakes_nobody_for_its_history(self):
        self._job_unbaselined("s1/subagents/sub-1a2b", "j2", exit_code=0)
        sid = "s1/subagents/sub-1a2b"
        self.assertFalse(jobs_on_disk.has_baseline(sid))
        jobs = scan_session(sid)
        jobs_on_disk.establish_baseline(sid, jobs)
        # Marked where the scan will look again — under the child, not the parent.
        self.assertTrue(jobs_on_disk.has_baseline(sid))
        self.assertFalse(jobs_on_disk.has_baseline("s1"))
        self.assertEqual(scan_session(sid), [])

    def test_a_child_job_ending_after_its_baseline_is_reported(self):
        sid = "s1/subagents/sub-1a2b"
        self._job_unbaselined(sid, "j2", pid=os.getpid(),
                              starttime=proc_starttime(os.getpid()))
        jobs_on_disk.establish_baseline(sid, scan_session(sid))
        self._job(sid, "j3", exit_code=0)
        self.assertEqual([j.job_key for j in scan_session(sid)], ["j2", "j3"])


class RearmTests(_ScanCase):
    def _worker(self, session_id: str = "s1") -> _RearmWorker:
        return _RearmWorker(session_id)

    def test_a_live_run_gets_its_watcher_back(self):
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()))
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

    def test_a_childs_live_run_is_re_armed_for_the_parent(self):
        """The parent picks up what its sub-agent left going.

        Indistinguishable from one of its own to whatever consumes the wake, which is
        this method's standing rule — except for the directory, which has to name the
        child's own, or the first poll resolves to nothing.
        """
        self._job("s1/subagents/sub-1a2b", "j2", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()))
        w = self._worker()
        result = w.rearm_detached_jobs()
        self.assertEqual(result["rearmed"], ["j2"])
        descriptor, owner = w.registered[0]
        self.assertEqual(owner, "s1", "the parent is the conversation to wake")
        self.assertTrue(descriptor["status_op"]["args"]["run_dir"]
                        .endswith("/subagents/sub-1a2b/jobs/j2"))

    def test_a_childs_ended_run_wakes_the_parent(self):
        self._job("s1/subagents/sub-1a2b", "j2", exit_code=0, command="make -j64")
        w = self._worker()
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["j2"])
        ev = self._events(w)[0]
        self.assertEqual(ev["session_id"], "s1")
        self.assertEqual(ev["state"], "done")
        self.assertEqual(ev["summary"]["command"], "make -j64")

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
                  starttime=proc_starttime(os.getpid()))
        w = self._worker()
        w.jobs._bg_jobs["j1"] = mock.Mock(task=mock.Mock(done=lambda: False))
        result = w.rearm_detached_jobs()
        self.assertEqual(result, {"rearmed": [], "reported": []})
        self.assertEqual(w.registered, [])

    def test_a_watcher_that_has_finished_does_not_count_as_holding_the_run(self):
        self._job("s1", "j1", exit_code=0)
        w = self._worker()
        w.jobs._bg_jobs["j1"] = mock.Mock(task=mock.Mock(done=lambda: True))
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["j1"])

    def test_several_runs_are_sorted_into_the_two_piles(self):
        self._job("s1", "live", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()))
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

    A connection arriving and a worker being built both ask, so the marker lives in the
    run's own directory and outlives whichever process wrote it.

    What the marker records is **delivery, not emission**. A run announced twice and
    taken in once is a nuisance the consumers dedup; a run settled at the moment its
    event was queued, with nothing yet reading the queue, is a wake nobody will ever be
    given — and every later scan agrees it was already paid. So an announcement repeats
    until a consumer has folded the wake into a turn, and stops for good once one has.
    """

    def test_a_reported_run_is_left_out_of_the_next_scan(self):
        self._job("s1", "j1", exit_code=0)
        job = scan_session("s1")[0]
        self.assertFalse(jobs_on_disk.was_reported(job))
        jobs_on_disk.mark_reported(job)
        self.assertTrue(jobs_on_disk.was_reported(job))
        self.assertEqual(scan_session("s1"), [])

    def test_the_marker_can_be_asked_for_anyway(self):
        self._job("s1", "j1", exit_code=0)
        jobs_on_disk.mark_reported(scan_session("s1")[0])
        self.assertEqual([j.job_key for j in
                          scan_session("s1", include_reported=True)], ["j1"])

    def test_a_live_run_is_never_filtered_by_the_marker(self):
        # Re-arming a run already held costs nothing — the worker dedups — whereas
        # filtering one out would leave it unwatched for good.
        self._job("s1", "j1", pid=os.getpid(),
                  starttime=proc_starttime(os.getpid()))
        jobs_on_disk.mark_reported(scan_session("s1")[0])
        self.assertEqual([j.job_key for j in scan_session("s1")], ["j1"])

    def test_the_worker_announces_a_run_until_something_takes_it_in(self):
        # Announcing is not delivering. The worker puts the event on its queue and
        # nothing has read it yet, so a rebuild must say it again: the alternative is a
        # wake that was emitted into an empty process and settled on the spot, which is
        # a conversation waiting for ever on a job that finished.
        self._job("s1", "j1", exit_code=0)
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["j1"])
        self.assertEqual(_RearmWorker("s1").rearm_detached_jobs()["reported"], ["j1"])

    def test_a_run_whose_wake_was_delivered_is_announced_no_more(self):
        # What settles a run: a consumer that folded its wake into a turn.
        self._job("s1", "j1", exit_code=0)
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs()["reported"], ["j1"])
        jobs_on_disk.mark_wakes_reported(self._events(w))
        self.assertEqual(_RearmWorker("s1").rearm_detached_jobs()["reported"], [])

    def test_an_announced_run_carries_what_the_wake_text_needs(self):
        # A run reported late must be indistinguishable from one reported on time: the
        # wake names the job's kind, which only the descriptor knows.
        self._job("s1", "j1", exit_code=0)
        w = _RearmWorker("s1")
        w.rearm_detached_jobs()
        ev = self._events(w)[0]
        self.assertEqual(ev["server"], "bash")
        self.assertEqual(ev["kind"], "shell-command")

    def test_a_marker_that_cannot_be_written_still_lets_the_run_be_announced(self):
        # Costing a duplicate wake is a nuisance; refusing to announce the run because
        # the marker failed is the bug this path exists to fix.
        self._job("s1", "j1", exit_code=0)
        w = _RearmWorker("s1")
        # Only the marker's write: patching `open` would break the scan that finds
        # the run in the first place, and then the test would pass for the wrong reason.
        with mock.patch("mimir.client.session.jobs_on_disk.os.makedirs",
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
        self.assertFalse(jobs_on_disk.has_baseline("s1"))
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs(), {"rearmed": [], "reported": []})
        self.assertTrue(jobs_on_disk.has_baseline("s1"))
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
                       "pid_starttime": proc_starttime(os.getpid())}, fh)
        w = _RearmWorker("s1")
        self.assertEqual(w.rearm_detached_jobs()["rearmed"], ["live"])
        self.assertTrue(jobs_on_disk.has_baseline("s1"))

    def test_the_baseline_marks_the_history_so_it_is_never_claimed_later(self):
        self._unbaselined_job("s1", "old", 0)
        _RearmWorker("s1").rearm_detached_jobs()
        jobs = scan_session("s1", include_reported=True)
        self.assertTrue(all(jobs_on_disk.was_reported(j) for j in jobs))

    def test_a_watcher_reporting_a_run_marks_it_so_a_restart_does_not_repeat_it(self):
        # The common case and the worse one: a job whose own watcher reported it
        # normally must not be woken again on the next attach.
        self._baseline("s1")
        self._unbaselined_job("s1", "j1", 0)
        jobs_on_disk.mark_job_reported("s1", "j1")
        self.assertEqual(_RearmWorker("s1").rearm_detached_jobs()["reported"], [])

    def test_marking_by_key_needs_neither_a_session_nor_a_job(self):
        jobs_on_disk.mark_job_reported(None, "j1")
        jobs_on_disk.mark_job_reported("s1", "")


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


class ARunWatchedByItsOwnServerTests(_ScanCase):
    """A proxy run is a Slurm job of this workspace *and* that server's state machine.

    It is filed among the Slurm descriptors so that "is anything watching this id" has
    one answer — otherwise the queue calls it unwatched and the remedy offered, adopting
    the id, puts a second watcher on a run that already has one. But the watcher put
    back after a reload must be *its* watcher: re-armed with the stock poll instead, the
    run would be held twice under two different keys, which no dedup catches, and one
    result would wake the conversation twice.
    """

    def test_the_scan_re_arms_the_watcher_the_run_recorded(self):
        self._slurm_job("s1", "20260101T000000Z-aaaa", "9001",
                        watcher=self._proxy_watcher())
        job, = scan_session("s1")
        self.assertTrue(job.live)
        self.assertEqual(job.job_key, "run-0001")
        self.assertEqual(job.descriptor()["server"], "proxy")
        self.assertEqual(job.status_op()["tool"], "proxy_eval_status")

    def test_an_ordinary_submission_still_gets_the_stock_poll(self):
        self._slurm_job("s1", "20260101T000000Z-bbbb", "9002")
        job, = scan_session("s1")
        self.assertEqual(job.job_key, "9002")
        self.assertEqual(job.descriptor()["server"], "hpc")
        self.assertEqual(job.status_op()["tool"], "slurm_job_status")

    def test_its_end_is_still_settled_by_the_slurm_record(self):
        """Liveness comes from the descriptor, not from the watcher that owns it."""
        self._slurm_job("s1", "20260101T000000Z-cccc", "9003", state="done",
                        watcher=self._proxy_watcher("run-0003"))
        job, = scan_session("s1")
        self.assertFalse(job.live)
        self.assertEqual(job.state, "done")
        self.assertEqual(job.descriptor()["server"], "proxy")

    def test_the_live_watcher_and_a_late_scan_settle_the_same_run(self):
        """The wake must not be delivered twice because two writers spelled it apart.

        A run reported by its own watcher while the window was open marks itself with
        the key and server the watcher knows. A scan finding the same run afterwards
        has to read that marker, or it announces a result the conversation already saw.
        """
        self._slurm_job("s1", "20260101T000000Z-dddd", "9004", state="done",
                        watcher=self._proxy_watcher("run-0004"))
        jobs_on_disk.mark_job_reported("s1", "run-0004", "proxy")
        job, = scan_session("s1", include_reported=True)
        self.assertTrue(jobs_on_disk.was_reported(job))

    def test_a_local_proxy_run_keeps_its_own_tree(self):
        """One that submitted nothing has no descriptor, and its marker stays put."""
        self._job("s1", "run-local")
        jobs_on_disk.mark_job_reported("s1", "run-local", "proxy")
        self.assertTrue(os.path.exists(os.path.join(
            self._tmp.name, "sessions", "s1", "jobs", "run-local", "reported")))
