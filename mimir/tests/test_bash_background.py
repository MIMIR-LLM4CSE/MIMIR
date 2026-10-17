"""Detached ``bash_run(background=True)`` jobs: launch, terminal state, stop.

The blocking path is capped, and no real build fits inside the cap. These cover the
part that makes a detached run usable rather than merely started: that a terminal state
is knowable afterwards (the exit code the shell records on its way out), that stopping
one takes the whole process group with it, and — the property that must not regress —
that detaching is a parameter on an already-validated command, not a way around the
validation. See ``_bash_jobs`` and ``server_bash.bash_run``.

``BlockingRunTests`` below covers the other half: a blocking call now goes through the
same launcher, so its output lives in a file. That is what lets a run stopped at the cap
still report what it printed, and lets the user detach one mid-flight without losing it.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_channel  # noqa: E402
import slurm_site  # noqa: E402

from mimir.tests.support.scheduler import local_slurm  # noqa: E402
import _bash_jobs  # noqa: E402
import server_bash  # noqa: E402

from mimir.servers._shared.trusted_read_roots import trusted_read_roots  # noqa: E402


def _jobs_on_disk() -> list[str]:
    """What the jobs root holds — a root that was never created holds nothing."""
    try:
        return os.listdir(_bash_jobs.jobs_root())
    except FileNotFoundError:
        return []


def _wait_terminal(job_key: str, timeout: float = 15.0) -> dict:
    """Poll until the job leaves 'running', the way the client watcher does."""
    deadline = time.time() + timeout
    payload = _bash_jobs.state(job_key)
    while payload["state"] == "running" and time.time() < deadline:
        time.sleep(0.1)
        payload = _bash_jobs.state(job_key)
    return payload


class BashBackgroundTests(unittest.TestCase):
    def setUp(self) -> None:
        # Jobs land under the session's own state dir; point the state dir at a temp
        # tree so a test run neither reads nor leaves state in the user's own. Set
        # through the environment rather than by patching a module constant, so what
        # the test exercises is the real resolution (state_paths.session_state_dir).
        self._tmp = tempfile.mkdtemp(prefix="mimir-bash-state-")
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._tmp

    def tearDown(self) -> None:
        for payload in _bash_jobs.listing():
            if payload["state"] == "running":
                _bash_jobs.stop(payload["job_key"])
        if self._orig_state is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_launch_returns_immediately_with_a_watchable_handle(self) -> None:
        # The point of the feature: the call comes back while the work is still going,
        # carrying the descriptor the client polls generically (no tool name in loop
        # code) rather than the output.
        started = time.time()
        result = server_bash.bash_run("sleep 2; echo done", background=True)
        self.assertEqual(result["status"], "ok")
        self.assertLess(time.time() - started, 1.5)

        job = result["background_job"]
        self.assertEqual(job["job_key"], result["job_key"])
        self.assertEqual(job["status_op"]["args"]["job_key"], result["job_key"])
        self.assertEqual(job["summary_op"]["args"]["op"], "output")
        # The ops must name tools this server actually exposes, or the watcher polls
        # into nothing and the run is never reported as finished.
        for op in (job["status_op"], job["summary_op"]):
            self.assertTrue(hasattr(server_bash, op["tool"]), op["tool"])

        self.assertEqual(_bash_jobs.state(result["job_key"])["state"], "running")
        self.assertEqual(_wait_terminal(result["job_key"])["state"], "done")

    def test_output_is_readable_while_the_job_is_still_running(self) -> None:
        # A two-hour build is only supervisable if its log can be read mid-flight.
        result = server_bash.bash_run("echo first; sleep 3", background=True)
        deadline = time.time() + 5
        seen = ""
        while time.time() < deadline and "first" not in seen:
            time.sleep(0.1)
            payload = server_bash.bash_job(op="output", job_key=result["job_key"])
            seen = payload.get("output", "")
        self.assertIn("first", seen)
        self.assertEqual(payload["state"], "running")

    def test_a_running_job_returns_a_short_tail_and_a_finished_one_the_full_log(self) -> None:
        # A mid-run read is a progress check: re-sending the whole log on each one is
        # what fed a copy loop. Once finished, the log is the result and comes back whole.
        cap = server_bash._RUNNING_OUTPUT
        result = server_bash.bash_run(
            f"seq 1 {cap}; sleep 3", background=True)   # ~5 bytes a line
        deadline = time.time() + 2.5
        running: dict = {}
        while time.time() < deadline and not running.get("truncated"):
            time.sleep(0.1)
            running = server_bash.bash_job(op="output", job_key=result["job_key"])
        self.assertEqual(running["state"], "running")
        self.assertTrue(running.get("truncated"))
        self.assertLessEqual(len(running["output"]), cap)
        _wait_terminal(result["job_key"])
        done = server_bash.bash_job(op="output", job_key=result["job_key"])
        self.assertGreater(len(done["output"]), 3 * cap)

    def test_exit_code_separates_done_from_crashed(self) -> None:
        for command, state, code in (
            ("echo ok", "done", 0),
            ("echo no; false", "crashed", 1),
            # An epilogue appended after the command is never reached by a script that
            # ends in `exit`, which reported an ordinary failure as 'unknown'. The EXIT
            # trap is what covers this path.
            ("echo bye; exit 3", "crashed", 3),
        ):
            with self.subTest(command=command):
                result = server_bash.bash_run(command, background=True)
                payload = _wait_terminal(result["job_key"])
                self.assertEqual(payload["state"], state)
                self.assertEqual(payload["returncode"], code)

    def test_stop_takes_the_whole_process_group(self) -> None:
        # Signalling the shell alone leaves the compiler it launched holding the
        # machine, while the job already reads as stopped.
        result = server_bash.bash_run("sleep 120 | cat", background=True)
        time.sleep(0.5)
        self.assertEqual(_bash_jobs.state(result["job_key"])["state"], "running")

        stopped = server_bash.bash_job_stop(job_key=result["job_key"])
        self.assertEqual(stopped["status"], "ok")
        self.assertIn(stopped["stopped"], ("sigterm", "sigkill"))
        survivors = subprocess.run(
            ["pgrep", "-g", str(result["pid"])], capture_output=True, text=True,
        ).stdout.split()
        self.assertEqual(survivors, [])

    def test_a_killed_job_reads_unknown_rather_than_crashed(self) -> None:
        # It returned no status, so there is none to act on: 'unknown' is the honest
        # answer, and folding it into 'crashed' would invent an exit code's meaning.
        result = server_bash.bash_run("sleep 120", background=True)
        time.sleep(0.5)
        server_bash.bash_job_stop(job_key=result["job_key"])
        self.assertEqual(_bash_jobs.state(result["job_key"])["state"], "unknown")

    def test_backgrounding_does_not_bypass_validation(self) -> None:
        # The whole guardrail rests on this: background is a parameter on a command
        # that passed the same checks, which is why '&' can stay refused.
        for command in (
            "sleep 5 &",                 # the operator detaching is still not the caller's
            "cat /etc/passwd",           # outside the workspace
            "bash -c 'sleep 5'",         # a shell nothing here can inspect
            "sbatch job.sh",             # cluster submission has its own handle
        ):
            with self.subTest(command=command):
                result = server_bash.bash_run(command, background=True)
                self.assertEqual(result["status"], "error", command)
                self.assertNotIn("job_key", result)

    def test_job_logs_live_under_a_trusted_read_root(self) -> None:
        # The model reads the log with the ordinary file tools while the run goes on;
        # that only works if the location is one the read servers and the policy gate
        # both already trust. The state dir is such a root, and is where jobs live now:
        # a fixed root under ~/.cache was outside every session, so one conversation
        # listed — and could kill — another's jobs, and nothing ever reclaimed the space.
        self.assertIn(self._tmp, trusted_read_roots())
        self.assertTrue(_bash_jobs.jobs_root().startswith(self._tmp))

    def test_jobs_of_one_session_are_invisible_to_another(self) -> None:
        # Scoped by resolving the root, not by filtering: each session's jobs sit in
        # its own directory, so there is nothing to filter and no way to name a job
        # that is not yours.
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            launched = server_bash.bash_run("sleep 30", background=True)["job_key"]
            self.assertIn(launched, [j["job_key"] for j in _bash_jobs.listing()])
            mine = _bash_jobs.jobs_root()
            with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-b"}):
                # Session B cannot see it, and so cannot reach it: the handle names no
                # job of B's, and A's run is untouched by B asking to stop it.
                self.assertEqual(_bash_jobs.listing(), [])
                self.assertNotEqual(_bash_jobs.jobs_root(), mine)
                self.assertEqual(_bash_jobs.state(launched)["error"], "No such job.")
                server_bash.bash_job_stop(job_key=launched)
            self.assertEqual(_bash_jobs.state(launched)["state"], "running")
            # Stopped from its own session, where it is addressable. Done inside the
            # patch: tearDown's sweep resolves the root too, and would not find it.
            self.assertEqual(server_bash.bash_job_stop(job_key=launched)["status"], "ok")

    def test_nothing_is_written_outside_the_state_dir(self) -> None:
        # The point of the move: MIMIR writes under its own state dir, the workspace,
        # and the /tmp scratchpad — never a cache root of its own in $HOME.
        legacy = os.path.expanduser("~/.cache/mimir_bash")
        before = os.path.isdir(legacy)
        job_key = server_bash.bash_run("echo hi", background=True)["job_key"]
        _wait_terminal(job_key)
        self.assertEqual(os.path.isdir(legacy), before)

    def test_an_unknown_handle_is_refused_with_a_way_forward(self) -> None:
        for job_key in ("", "../../etc", "nope"):
            with self.subTest(job_key=job_key):
                payload = server_bash.bash_job(job_key=job_key)
                self.assertEqual(payload["status"], "error")
        self.assertEqual(server_bash.bash_job(op="nonsense")["status"], "error")

    def test_a_well_formed_handle_naming_nothing_is_an_error_not_an_ok(self) -> None:
        """``valid_key`` checks the shape of a handle, not that it names anything.

        So a well-formed key for a job this host does not have reached the job layer and
        came back carrying ``error`` — which was then wrapped in ``ok()``, producing a
        payload that said both at once. A caller reading ``status`` first was told the
        stop had succeeded while nothing was stopped. No longer hypothetical: job
        directories are per session, so a handle from another conversation is exactly a
        well-formed key naming nothing here.
        """
        absent = "20200101T000000Z-abcd"
        self.assertTrue(_bash_jobs.valid_key(absent))
        for payload in (server_bash.bash_job(op="status", job_key=absent),
                        server_bash.bash_job(op="output", job_key=absent),
                        server_bash.bash_job_stop(job_key=absent)):
            self.assertEqual(payload["status"], "error")
            self.assertIn("No such job", payload["error"])
            self.assertIn("op='list'", payload.get("hint", ""))

    def test_a_redirected_job_says_where_its_output_went(self) -> None:
        # A command with its own redirect writes nothing to the job's log, so the
        # completion summary would read as "finished, said nothing" for a run that
        # wrote everything elsewhere. Observed on the TPL build, whose command carried
        # '> tpl_install.log 2>&1'. The target is named, not merely alluded to.
        result = server_bash.bash_run("echo hidden > out.txt", background=True)
        _wait_terminal(result["job_key"])
        payload = server_bash.bash_job(op="output", job_key=result["job_key"])
        self.assertEqual(payload["output"], "")
        # Under 'note': 'hint' is reserved to error payloads and ok() strips it, so a
        # message put there is silently lost — which is what this first caught.
        self.assertIn("out.txt", payload["note"])
        os.remove(os.path.join(server_bash._WORKSPACE_ROOT, "out.txt"))

    def test_a_quiet_job_is_not_annotated_for_being_quiet(self) -> None:
        # Long jobs that legitimately print nothing — an extraction, a copy, a quiet
        # install — are the ordinary case, and must not each carry a caveat about
        # output that might have gone elsewhere. Nothing was redirected, so there is
        # nothing to say.
        target = os.path.join(server_bash._WORKSPACE_ROOT, "quiet-job-dir")
        result = server_bash.bash_run("mkdir -p quiet-job-dir", background=True)
        _wait_terminal(result["job_key"])
        payload = server_bash.bash_job(op="output", job_key=result["job_key"])
        self.assertEqual(payload["output"], "")
        self.assertNotIn("note", payload)
        os.rmdir(target)

    def test_a_job_that_printed_something_carries_no_note(self) -> None:
        result = server_bash.bash_run("echo spoken", background=True)
        _wait_terminal(result["job_key"])
        payload = server_bash.bash_job(op="output", job_key=result["job_key"])
        self.assertIn("spoken", payload["output"])
        self.assertNotIn("note", payload)

    def test_listing_reports_every_job_newest_first(self) -> None:
        first = server_bash.bash_run("echo one", background=True)["job_key"]
        time.sleep(1.1)  # keys carry a one-second stamp; make the order unambiguous
        second = server_bash.bash_run("echo two", background=True)["job_key"]
        keys = [j["job_key"] for j in server_bash.bash_job(op="list")["jobs"]]
        self.assertEqual(keys[:2], [second, first])


class BlockingRunTests(unittest.TestCase):
    """The blocking path, which runs a job and waits on it.

    Same temp jobs root as above, plus a temp state dir: the divert channel is a file
    the client and the server both reach, and a test must not write into the user's.
    """

    def setUp(self) -> None:
        # One temp state dir now covers both: the job directories AND the divert
        # channel live under it, which is the point — they belong to the same session.
        self._state = tempfile.mkdtemp(prefix="mimir-bash-state-")
        self._tmp = self._state
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._state

    def tearDown(self) -> None:
        for payload in _bash_jobs.listing():
            if payload["state"] == "running":
                _bash_jobs.stop(payload["job_key"])
        if self._orig_state is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_state
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._state, ignore_errors=True)

    def _await_current(self, timeout: float = 10.0) -> dict:
        """The run the server says it is waiting on, once it says so."""
        import json
        path = os.path.join(run_channel._dir("bash_run"), "current.json")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with open(path) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                time.sleep(0.02)
        self.fail("the server never announced a foreground run")

    def test_a_timeout_returns_the_output_produced_before_the_kill(self) -> None:
        # The headline property. A timeout that kills the process group and throws away
        # everything it wrote buys nothing for the whole wait, and the model re-runs from
        # scratch. The kill is what bounds a command gone astray; the bytes come back
        # with it.
        captured = {}
        watcher = threading.Thread(
            target=lambda: captured.update(self._await_current()), daemon=True)
        watcher.start()
        result = server_bash.bash_run("echo early; sleep 30", timeout=1)
        watcher.join(timeout=5)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["partial"])
        self.assertIn("early", result["stdout"])
        self.assertIn("timed out", result["error"])

        # And it really is dead: the whole group, not just the shell.
        pid = captured.get("pid")
        self.assertIsNotNone(pid, "never saw which process was running")
        survivors = subprocess.run(["pgrep", "-g", str(pid)],
                                   capture_output=True, text=True).stdout.strip()
        self.assertEqual(survivors, "")

    def test_a_timeout_leaves_no_job_directory(self) -> None:
        server_bash.bash_run("sleep 30", timeout=1)
        self.assertEqual(_jobs_on_disk(), [])

    def test_a_completed_blocking_run_leaves_no_job_directory(self) -> None:
        # A job dir per `ls` would grow the cache without anyone holding a handle.
        server_bash.bash_run("echo hi")
        self.assertEqual(_jobs_on_disk(), [])

    def test_a_refused_command_creates_no_job(self) -> None:
        # The blocking mirror of test_backgrounding_does_not_bypass_validation:
        # validation runs before anything is launched, on both paths.
        for command in ("echo hi & echo there", "bash -c 'echo hi'", "cat /etc/passwd"):
            with self.subTest(command=command):
                result = server_bash.bash_run(command, timeout=5)
                self.assertEqual(result["status"], "error")
        self.assertEqual(_jobs_on_disk(), [])

    def test_foreground_stderr_stays_separate_from_stdout(self) -> None:
        # The job launcher merges the two by default, which a tailed log wants and a
        # blocking payload must not have: the failure classifier reads stderr alone,
        # and the UI gives it its own pane.
        result = server_bash.bash_run("echo out; echo bad 1>&2")
        self.assertEqual(result["stdout"], "out\n")
        self.assertEqual(result["stderr"], "bad\n")

    def test_foreground_reports_the_real_exit_code(self) -> None:
        # Read from waitpid, not from the trap file: a command installing its own EXIT
        # trap replaces the one the launcher wrote, and the last case here is exactly
        # the one that file cannot answer.
        for command, expected in (
            ("echo ok", 0),
            ("echo no; false", 1),
            ("echo bye; exit 3", 3),
            ("trap 'echo leaving' EXIT; exit 3", 3),
        ):
            with self.subTest(command=command):
                self.assertEqual(server_bash.bash_run(command)["returncode"], expected)

    def test_a_divert_moves_a_running_run_to_the_background(self) -> None:
        # Nothing is killed and nothing is lost: the caller gets what the run had
        # printed so far plus the ordinary handle, and the run carries on to its end.
        def divert() -> None:
            current = self._await_current()
            with open(os.path.join(run_channel._dir("bash_run"), "divert"), "w") as fh:
                fh.write(current["job_key"])

        threading.Thread(target=divert, daemon=True).start()
        started = time.time()
        result = server_bash.bash_run("echo early; sleep 4; echo late", timeout=60)

        self.assertLess(time.time() - started, 3.0, "it kept blocking")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reason"], "diverted")
        self.assertTrue(result["partial"])
        self.assertIn("early", result["stdout"])
        # No exit code: the run has not produced one, and inventing it is the one lie
        # the caller could not detect.
        self.assertNotIn("returncode", result)
        job_key = result["job_key"]
        self.assertEqual(result["background_job"]["job_key"], job_key)
        self.assertEqual(_bash_jobs.state(job_key)["state"], "running")

        self.assertEqual(_wait_terminal(job_key)["state"], "done")
        with open(_bash_jobs.log_path(job_key)) as fh:
            self.assertIn("late", fh.read())

    def test_a_diverted_run_keeps_its_job_directory(self) -> None:
        def divert() -> None:
            current = self._await_current()
            with open(os.path.join(run_channel._dir("bash_run"), "divert"), "w") as fh:
                fh.write(current["job_key"])

        threading.Thread(target=divert, daemon=True).start()
        result = server_bash.bash_run("sleep 4", timeout=60)
        self.assertIn(result["job_key"], _jobs_on_disk())

    def test_a_divert_naming_another_run_is_not_consumed(self) -> None:
        # Two clients share one state dir. A request must name the run it meant.
        def divert() -> None:
            self._await_current()
            with open(os.path.join(run_channel._dir("bash_run"), "divert"), "w") as fh:
                fh.write("20200101T000000Z-dead")

        threading.Thread(target=divert, daemon=True).start()
        result = server_bash.bash_run("echo hi; sleep 1", timeout=30)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("background_job", result)


class FinishedOutputClipTests(unittest.TestCase):
    """A long finished run keeps its beginning and its end, not just the beginning.

    With the head alone, a build's last lines (the final error, the linked target) were
    cut, and the model piped every build into ``tail`` to see them — which holds the
    output back until the build ends and hides its progress from the user.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "run.log")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, lines: int) -> None:
        with open(self.path, "w") as fh:
            for i in range(lines):
                fh.write(f"[{i:5d}] Building CXX object obj{i}.o\n")

    def test_a_long_output_keeps_its_first_and_last_lines(self) -> None:
        self._write(5000)
        text, cut = _bash_jobs._clip(self.path, 4096, "ends")
        self.assertTrue(cut)
        lines = text.splitlines()
        self.assertEqual(lines[0], "[    0] Building CXX object obj0.o")
        self.assertEqual(lines[-1], "[ 4999] Building CXX object obj4999.o")
        self.assertIn("bytes of output omitted", text)
        # Whole lines only on both sides of the marker.
        for line in lines:
            if line and "omitted" not in line:
                self.assertRegex(line, r"^\[\s*\d+\] Building CXX object obj\d+\.o$")
        self.assertLess(len(text), 4096 + 100)

    def test_a_short_output_is_returned_whole(self) -> None:
        self._write(10)
        text, cut = _bash_jobs._clip(self.path, 4096, "ends")
        self.assertFalse(cut)
        self.assertEqual(len(text.splitlines()), 10)


class RewatchHandleTests(unittest.TestCase):
    """Asking a job's state hands back the handle to track it with.

    The watcher that was holding a run dies with the agent that made it, which an
    editor window reload is enough to cause; the detached process carries on in its
    own session directory regardless. So the state op answers with the handle as well
    as the state, and asking becomes the way to pick tracking back up — without
    re-arming runs nobody asked about.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-bash-rewatch-")
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._tmp

    def tearDown(self) -> None:
        for payload in _bash_jobs.listing():
            if payload["state"] == "running":
                _bash_jobs.stop(payload["job_key"])
        if self._orig_state is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _launch(self, command: str = "sleep 30") -> str:
        return server_bash.bash_run(command, background=True)["job_key"]

    def test_the_status_of_a_running_job_carries_a_handle(self) -> None:
        key = self._launch()
        status = server_bash.bash_job(op="status", job_key=key)
        self.assertEqual(status["state"], "running")
        handles = status["background_jobs"]
        self.assertEqual([h["job_key"] for h in handles], [key])
        self.assertEqual(handles[0]["status_op"]["args"]["job_key"], key)
        self.assertEqual(handles[0]["summary_op"]["args"]["op"], "output")

    def test_the_plural_key_is_used_so_a_read_is_not_mistaken_for_a_launch(self) -> None:
        """``background_job`` would make the row a run and, with no watcher, a wait."""
        key = self._launch()
        status = server_bash.bash_job(op="status", job_key=key)
        self.assertNotIn("background_job", status)

    def test_a_finished_job_carries_none(self) -> None:
        key = self._launch("true")
        _wait_terminal(key)
        status = server_bash.bash_job(op="status", job_key=key)
        self.assertNotEqual(status["state"], "running")
        self.assertNotIn("background_jobs", status)

    def test_a_listing_carries_one_handle_per_running_job(self) -> None:
        live, done = self._launch(), self._launch("true")
        _wait_terminal(done)
        listing = server_bash.bash_job(op="list")
        self.assertEqual([h["job_key"] for h in listing["background_jobs"]], [live])

    def test_a_listing_with_nothing_running_carries_none(self) -> None:
        _wait_terminal(self._launch("true"))
        self.assertNotIn("background_jobs", server_bash.bash_job(op="list"))

    def test_a_handle_names_the_jobs_own_directory(self) -> None:
        key = self._launch()
        handle = server_bash.bash_job(op="status", job_key=key)["background_jobs"][0]
        self.assertTrue(handle["run_dir"].endswith(key))
        self.assertTrue(os.path.isdir(handle["run_dir"]))

    def test_reading_a_running_jobs_log_re_arms_it_too(self) -> None:
        """Reading the log mid-run is as much a "where is it at?" as asking the state."""
        key = self._launch()
        out = server_bash.bash_job(op="output", job_key=key)
        self.assertEqual([h["job_key"] for h in out["background_jobs"]], [key])

    def test_the_log_of_a_finished_job_carries_no_handle(self) -> None:
        key = self._launch("true")
        _wait_terminal(key)
        self.assertNotIn("background_jobs", server_bash.bash_job(op="output", job_key=key))


class AdoptingARunningProcessTests(unittest.TestCase):
    """``bash_watch(pid)`` — the counterpart of a launch for work already going.

    A process something else started is watched by nothing: tracking is the record a
    launch writes, and nothing wrote one. So nobody is told when it ends, and it does
    not keep this workspace's server alive while it runs — which is how a night goes
    to a run that finished at two in the morning with no one listening.

    What it cannot give is a status. The exit-code trap is installed by the launch and
    cannot be added to a process already running, so the end of an adopted job is
    ``unknown`` — honestly, rather than a ``done`` nobody observed.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-adopt-state-")
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._tmp
        self._procs: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if self._orig_state is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_state
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sleeper(self) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self._procs.append(proc)
        return proc

    def test_a_running_process_becomes_a_watchable_job(self) -> None:
        proc = self._sleeper()
        res = server_bash.bash_watch(pid=proc.pid, what="the overnight solver")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["state"], "running")
        # The handle the client watcher arms on. Plural, as every answer about work
        # this call did not launch is: an adoption is read without the launch cap.
        handle = res["background_jobs"][0]
        self.assertEqual(handle["status_op"]["tool"], "bash_job")
        self.assertEqual(handle["job_key"], res["job_key"])
        self.assertNotIn("background_job", res)
        # And the state the watcher polls, through the ordinary seam.
        self.assertEqual(_bash_jobs.state(res["job_key"])["state"], "running")
        # What was adopted, not what will happen next: a resume is the client's to
        # promise, and only once it is holding a watcher.
        self.assertNotIn("resumed", res["headline"])
        self.assertIn(str(proc.pid), res["headline"])

    def test_what_it_is_is_kept_as_the_command(self) -> None:
        proc = self._sleeper()
        res = server_bash.bash_watch(pid=proc.pid, what="the overnight solver")
        self.assertEqual(_bash_jobs.state(res["job_key"])["command"],
                         "the overnight solver")

    def test_the_end_of_an_adopted_run_is_unknown_and_says_why(self) -> None:
        proc = self._sleeper()
        res = server_bash.bash_watch(pid=proc.pid)
        proc.kill()
        proc.wait()
        payload = _wait_terminal(res["job_key"])
        self.assertEqual(payload["state"], "unknown")
        # Not "it was killed from outside": an adopted process never had the trap, so
        # the absence of a status says nothing about how it ended.
        self.assertIn("adopted rather than started here", payload["note"])

    def test_a_pid_that_is_not_running_is_refused(self) -> None:
        proc = self._sleeper()
        pid = proc.pid
        proc.kill()
        proc.wait()
        res = server_bash.bash_watch(pid=pid)
        self.assertEqual(res["status"], "error")
        self.assertEqual(_jobs_on_disk(), [], "it recorded a job anyway")

    def test_a_nonsense_pid_is_refused_before_anything_is_written(self) -> None:
        for bad in (0, -1, "nope"):
            with self.subTest(pid=bad):
                self.assertEqual(server_bash.bash_watch(pid=bad)["status"], "error")
        self.assertEqual(_jobs_on_disk(), [])

    def test_the_scan_finds_an_adopted_job_the_way_it_finds_a_launched_one(self) -> None:
        # The whole point: the client's scan reads descriptors off disk, and this is
        # what makes an outside process appear in it — live, so a watcher is armed and
        # the process stays alive for it.
        from mimir.client.session import jobs_on_disk
        proc = self._sleeper()
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s-adopt"}):
            res = server_bash.bash_watch(pid=proc.pid, what="solver")
        with patch("mimir.client.config.paths.STATE_DIR", self._tmp):
            jobs = jobs_on_disk.scan_session("s-adopt")
        self.assertEqual([(j.job_key, j.kind, j.live) for j in jobs],
                         [(res["job_key"], "shell", True)])

    def test_where_its_output_goes_is_recorded_for_the_reader(self) -> None:
        # An adopted job has no log of its own, so an empty one must not read as a run
        # that printed nothing.
        proc = self._sleeper()
        target = os.path.join(self._tmp, "solver.log")
        res = server_bash.bash_watch(pid=proc.pid, log=target)
        payload = _bash_jobs.output(res["job_key"], 4096)
        self.assertIn(target, payload.get("note", ""))


class AParentPollsItsChildsJobTests(unittest.TestCase):
    """A run a sub-agent launched, read by the conversation that delegated to it.

    A child's session is stored under its parent's, so its jobs are real runs of this
    conversation's work that do not resolve from a bare handle. They must resolve some
    way, and not for convenience: a handle resolved against the wrong session reads as
    "No such job", the job layer spells that ``unknown``, and ``unknown`` is terminal.
    A parent that adopted a run on a bare handle would announce a healthy build as lost
    on its first poll.
    """

    _KEY = "20261013T101500Z-ab12"

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-bash-child-")
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        self._orig_session = os.environ.get("MIMIR_SESSION_ID")
        os.environ["MIMIR_STATE_DIR"] = self._tmp
        os.environ["MIMIR_SESSION_ID"] = "parent-abc"
        self._child_dir = os.path.join(
            self._tmp, "sessions", "parent-abc", "subagents", "sub-1a2b3c4d",
            "jobs", self._KEY)
        os.makedirs(self._child_dir)
        import json
        json.dump({"job_key": self._KEY, "command": "make -j64", "pid": os.getpid(),
                   "pid_starttime": None, "started_at": time.time()},
                  open(os.path.join(self._child_dir, "meta.json"), "w"))
        with open(os.path.join(self._child_dir, "run.log"), "w") as fh:
            fh.write("linking...\n")

    def tearDown(self) -> None:
        for name, value in (("MIMIR_STATE_DIR", self._orig_state),
                            ("MIMIR_SESSION_ID", self._orig_session)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_a_bare_handle_reads_a_healthy_run_as_terminal(self) -> None:
        """Why the directory has to travel: the wrong answer here is not an error.

        It is ``unknown``, which the watcher treats as the end of the run — so the
        failure mode is a wake saying a build can no longer be tracked while it is
        still linking.
        """
        answer = server_bash.bash_job(op="status", job_key=self._KEY)
        self.assertEqual(answer["state"], "unknown")

    def test_the_directory_from_the_descriptor_resolves_it(self) -> None:
        answer = server_bash.bash_job(op="status", job_key=self._KEY,
                                      run_dir=self._child_dir)
        self.assertEqual(answer["status"], "ok")
        self.assertEqual(answer["state"], "running")
        self.assertEqual(answer["command"], "make -j64")

    def test_the_handle_it_hands_back_names_the_childs_own_directory(self) -> None:
        """Not one derived from the reading session.

        The answer to a status read is also how tracking is picked back up, so a handle
        pointing at the parent's own jobs directory would resolve to nothing on the
        very next poll.
        """
        answer = server_bash.bash_job(op="status", job_key=self._KEY,
                                      run_dir=self._child_dir)
        handle = answer["background_jobs"][0]
        self.assertEqual(handle["run_dir"], self._child_dir)

    def test_its_log_is_readable_through_the_same_directory(self) -> None:
        answer = server_bash.bash_job(op="output", job_key=self._KEY,
                                      run_dir=self._child_dir)
        self.assertEqual(answer["status"], "ok")
        self.assertIn("linking...", answer["output"])

    def test_another_conversations_job_is_not_this_ones_to_read(self) -> None:
        """Containment is the whole rule, and it needs no new concept.

        A directory under this session grants this session's runs and its children's;
        a sibling conversation's is not under it, however well formed.
        """
        other = os.path.join(self._tmp, "sessions", "other-xyz", "jobs", self._KEY)
        os.makedirs(other)
        answer = server_bash.bash_job(op="status", job_key=self._KEY, run_dir=other)
        self.assertEqual(answer["status"], "error")
        self.assertIn("outside this session", answer["error"])

    def test_a_directory_that_is_not_a_job_is_refused(self) -> None:
        for path in (os.path.join(self._tmp, "sessions", "parent-abc"),
                     os.path.join(self._tmp, "sessions", "parent-abc", "jobs", "nope")):
            with self.subTest(path=path):
                answer = server_bash.bash_job(op="status", job_key=self._KEY,
                                              run_dir=path)
                self.assertEqual(answer["status"], "error")

    def test_a_directory_naming_another_job_than_the_handle_is_refused(self) -> None:
        """The pair travels together on the descriptor, so a mismatch is a composed
        argument rather than a handover — and composing one is how a reader ends up
        reading a run it did not mean to."""
        sibling = os.path.join(os.path.dirname(self._child_dir),
                               "20261013T101500Z-cd34")
        os.makedirs(sibling)
        answer = server_bash.bash_job(op="status", job_key=self._KEY, run_dir=sibling)
        self.assertEqual(answer["status"], "error")
        self.assertIn("different job", answer["error"])


class LearnedSchedulersTests(unittest.TestCase):
    """Which schedulers a workspace reaches, learned from calls it was making anyway.

    On a login node whose Slurm tools live on another host, every typed cluster tool
    answers "sbatch is not available" — so the work goes out as `ssh host "sbatch …"`,
    which writes no descriptor: nothing watches the job, no wake is owed when it ends,
    and the server stops on its idle timer with the run still going. The fact the typed
    tools were missing is the host, and the model names it constantly while reading the
    queue. One successful call is enough, and a setting nobody knows to set before the
    first lost night is the wrong mechanism.

    An inventory rather than an answer: a machine reaches its own scheduler and
    whatever ``ssh`` gets to, so learning one *adds* a destination and redirects
    nothing.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-schedulers-")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        patcher = patch.dict(os.environ, {"MIMIR_STATE_DIR": self._tmp}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        # A host without Slurm, which is the arrangement this learning is for. The
        # probe is memoised in the module, so it is set rather than mocked per call,
        # and restored — a developer's own machine may have sbatch.
        self._local(False)

    def _local(self, present: bool) -> None:
        """Answer the local-Slurm probe with *present* for the rest of this test."""
        local_slurm(self, present)

    def _ran(self, argv: list[str], *, ok: bool = True,
             queued: list[str] | None = None) -> dict:
        """Run one parsed command past the learner, and answer with its payload."""
        result = {"status": "ok" if ok else "error", "returncode": 0 if ok else 1}
        orig = server_bash.slurm_queued_by_host
        # Every reachable scheduler is asked, and each answers for itself; the tests
        # here have one host in play, so one entry says it.
        server_bash.slurm_queued_by_host = lambda *_a, **_k: {
            host: queued for host in (slurm_site.hosts() or [""])}
        server_bash._QUEUE_ASKED_AT = 0.0
        try:
            server_bash._note_scheduler([argv], result)
        finally:
            server_bash.slurm_queued_by_host = orig
        return result

    def _host_after(self, argv: list[str], *, ok: bool = True) -> str:
        """The one remote scheduler learned, or "" — the shape most cases assert."""
        self._ran(argv, ok=ok)
        remote = [host for host in slurm_site.hosts() if host]
        return remote[0] if len(remote) == 1 else ",".join(remote)

    def test_a_successful_remote_queue_read_records_the_host(self) -> None:
        self.assertEqual(
            self._host_after(["ssh", "host-a", "squeue -u $USER -o '%.10i'"]),
            "host-a")

    def test_a_failed_call_proves_nothing(self) -> None:
        # The far side may not have had the tools, or the host may not exist. Only a
        # command that answered says where the scheduler is.
        self.assertEqual(self._host_after(["ssh", "host-a", "squeue"], ok=False), "")

    def test_a_remote_command_that_is_not_slurms_records_nothing(self) -> None:
        self.assertEqual(self._host_after(["ssh", "host-a", "tail -2 run.log"]), "")

    def test_the_host_is_found_past_ssh_flags(self) -> None:
        self.assertEqual(
            self._host_after(["ssh", "-n", "-p", "2222", "me@host-a", "sacct -j 1"]),
            "me@host-a")

    def test_a_local_command_mentioning_a_scheduler_records_nothing(self) -> None:
        # The difference between a fact and a guess over the text: this command
        # contains the word, and says nothing about where a scheduler is.
        self.assertEqual(self._host_after(["grep", "-r", "sbatch", "notes/"]), "")

    def test_it_is_remembered_across_calls_and_written_once(self) -> None:
        self._ran(["ssh", "host-a", "squeue"])
        path = os.path.join(slurm_site.state_dir(), "slurm_hosts")
        before = os.stat(path).st_mtime_ns
        self._ran(["ssh", "host-a", "sinfo"])
        self.assertEqual(os.stat(path).st_mtime_ns, before,
                         "a scheduler already known is re-recorded on every queue "
                         "listing otherwise")
        self.assertEqual(slurm_site.hosts(), ["host-a"])

    def test_another_host_is_added_rather_than_substituted(self) -> None:
        """One conversation, two clusters — a reference run on one, a sweep on the other.

        The second must not evict the first: the jobs already running on it are polled
        through this same inventory, and a scheduler dropped from it is a run whose end
        nothing is waiting for.
        """
        self._ran(["ssh", "host-a", "squeue"])
        self._ran(["ssh", "host-b", "sacct -j 1"])
        self.assertEqual(slurm_site.hosts(), ["host-a", "host-b"])

    def test_a_host_with_its_own_slurm_keeps_it_first(self) -> None:
        """A machine with a scheduler has it whether anything was learned or not.

        It is a probe, not a record — true before any call is made — and it comes
        first because it is the one whose filesystems the work is already on. An ssh
        elsewhere adds a second destination and takes nothing away: the submissions
        that belong here keep a scheduler to go to.
        """
        self._local(True)
        self._ran(["ssh", "host-b", "squeue -u $USER"])
        self.assertEqual(slurm_site.hosts(), ["", "host-b"])

    def test_the_sessions_of_one_workspace_share_the_schedulers(self) -> None:
        """The inventory belongs to the work, not to the conversation doing it.

        Two conversations in one workspace are two conversations on one project, whose
        inputs are on one set of filesystems and whose jobs therefore run on one set of
        clusters. A scheduler the first proved reachable is one the second can submit
        to, and it must not have to prove it again.

        What makes it load-bearing rather than a convenience: a job's descriptor is
        already read across every session of the workspace (``recorded_jobs``), so the
        second conversation can find a job the first submitted. Learned per session,
        the registry would hand it that job with no scheduler to poll it on — a run
        nothing is watching, which is the failure this whole module exists against.
        """
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "sess-local"}, clear=False):
            self._ran(["ssh", "host-a", "squeue"])
            self.assertEqual(slurm_site.hosts(), ["host-a"])
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "sess-other"}, clear=False):
            self.assertEqual(slurm_site.hosts(), ["host-a"],
                             "a sibling's scheduler is reachable before this "
                             "conversation has made a call of its own")
            self._ran(["ssh", "host-b", "squeue"])
            self.assertEqual(slurm_site.hosts(), ["host-a", "host-b"])
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "sess-local"}, clear=False):
            self.assertEqual(slurm_site.hosts(), ["host-a", "host-b"],
                             "and the learning goes both ways")


    def test_a_named_host_is_taken_as_given(self) -> None:
        """The registry is a convenience, not an allowlist.

        A host the user names is one they can already ``ssh`` to, and refusing it for
        not having been learned yet would make the learning a gate rather than a
        shortcut. It joins the inventory when a command actually succeeds there.
        """
        host, refusal = slurm_site.resolve_host("never-seen")
        self.assertEqual(host, "never-seen")
        self.assertEqual(refusal, "")

    def test_naming_local_where_there_is_no_local_slurm_is_refused(self) -> None:
        # Said rather than silently sent over ssh to nothing: "local" means this host,
        # and this host has no scheduler.
        self._ran(["ssh", "host-a", "squeue"])
        host, refusal = slurm_site.resolve_host("local")
        self.assertEqual(host, "")
        self.assertIn("no local scheduler", refusal)
        self.assertIn("host-a", refusal)

    def test_nothing_reachable_says_how_to_reach_something(self) -> None:
        host, refusal = slurm_site.resolve_host()
        self.assertEqual(host, "")
        self.assertIn("squeue -u $USER", refusal)
        self.assertIn("slurm_schedulers", refusal)

    def test_a_choice_between_schedulers_is_not_made_here(self) -> None:
        """What the refusal has to carry, since it is the model's whole prompt to act.

        The candidates, so it can choose; what the choice follows, so it chooses on
        the work rather than on order; and that the user is who settles it otherwise.
        """
        self._local(True)
        self._ran(["ssh", "host-b", "squeue"])
        host, refusal = slurm_site.resolve_host()
        self.assertEqual(host, "")
        self.assertIn("local, host-b", refusal)
        self.assertIn("GPUs", refusal)
        self.assertIn("ask them", refusal)

    def test_a_command_that_reached_the_scheduler_answers_with_the_gap(self):
        """The case the whole path exists for, closed where it actually happens.

        A job submitted through a site launcher is tracked by nothing, and neither the
        launcher call nor the ``squeue`` that checks it a second later reaches a typed
        cluster tool — so the reminder that fires when a turn concludes with unwatched
        work never got its fact. A command that just *proved* the scheduler answers is
        where that fact is free: the ids it is running, minus the ones a descriptor
        covers. Carried on this payload, which is all it takes — the client's observer
        reads the field, not the tool name.
        """
        result = self._ran(["squeue", "-u", "me"],
                           queued=["15306620", "15306621_1", "15306621_2"])
        # By array, because adoption is by array: two task rows are one descriptor.
        self.assertEqual(result["unwatched"], ["15306620", "15306621"])
        self.assertIn("slurm_watch", result["unwatched_hint"])

    def test_a_remote_queue_read_answers_with_it_too(self):
        result = self._ran(["ssh", "host-a", "squeue -u me"], queued=["99"])
        self.assertEqual(result["unwatched"], ["99"])

    def test_a_command_that_never_touched_the_scheduler_asks_nothing(self):
        # The trigger is the validator's parse, not a word in the text: this one reads
        # a log, and asking the scheduler on every shell command would be a queue read
        # per call for the life of the session.
        result = self._ran(["tail", "-5", "run.log"], queued=["99"])
        self.assertNotIn("unwatched", result)

    def test_a_scheduler_that_will_not_answer_asserts_nothing(self):
        # None is "unknown", and unknown must never read as "nothing is running" —
        # silence taken for an empty queue is the whole failure this exists for.
        result = self._ran(["squeue"], queued=None)
        self.assertNotIn("unwatched", result)

    def test_the_queue_is_read_once_in_a_while_not_once_per_call(self):
        # A turn reads the queue several times — after a submission, then again for
        # progress — and the reminder needs the answer once.
        first = self._ran(["squeue"], queued=["99"])
        self.assertEqual(first["unwatched"], ["99"])
        second = {"status": "ok", "returncode": 0}
        orig = server_bash.slurm_queued_by_host
        server_bash.slurm_queued_by_host = lambda *_a, **_k: {"": ["99"]}
        try:
            server_bash._note_scheduler([["squeue"]], second)
        finally:
            server_bash.slurm_queued_by_host = orig
        self.assertNotIn("unwatched", second)


if __name__ == "__main__":
    unittest.main()
