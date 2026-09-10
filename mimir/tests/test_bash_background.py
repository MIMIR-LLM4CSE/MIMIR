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
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _bash_divert  # noqa: E402
import _bash_jobs  # noqa: E402
import server_bash  # noqa: E402

from mimir.servers._shared.trusted_read_roots import TRUSTED_CACHE_ROOTS  # noqa: E402


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
        # Jobs land in a real cache dir; point it at a temp tree so a test run neither
        # reads nor leaves state in the user's own.
        self._tmp = tempfile.mkdtemp(prefix="mimir-bash-jobs-")
        self._orig_root = _bash_jobs.JOBS_ROOT
        _bash_jobs.JOBS_ROOT = self._tmp

    def tearDown(self) -> None:
        for payload in _bash_jobs.listing():
            if payload["state"] == "running":
                _bash_jobs.stop(payload["job_key"])
        _bash_jobs.JOBS_ROOT = self._orig_root
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
        # both already trust.
        self.assertTrue(
            any(root.rstrip("/").endswith("mimir_bash") for root in TRUSTED_CACHE_ROOTS),
            TRUSTED_CACHE_ROOTS,
        )
        self.assertTrue(self._orig_root.startswith(
            os.path.expanduser("~/.cache/mimir_bash")))

    def test_an_unknown_handle_is_refused_with_a_way_forward(self) -> None:
        for job_key in ("", "../../etc", "nope"):
            with self.subTest(job_key=job_key):
                payload = server_bash.bash_job(job_key=job_key)
                self.assertEqual(payload["status"], "error")
        self.assertEqual(server_bash.bash_job(op="nonsense")["status"], "error")

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


if __name__ == "__main__":
    unittest.main()


class BlockingRunTests(unittest.TestCase):
    """The blocking path, which runs a job and waits on it.

    Same temp jobs root as above, plus a temp state dir: the divert channel is a file
    the client and the server both reach, and a test must not write into the user's.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-bash-jobs-")
        self._orig_root = _bash_jobs.JOBS_ROOT
        _bash_jobs.JOBS_ROOT = self._tmp
        self._state = tempfile.mkdtemp(prefix="mimir-bash-state-")
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._state

    def tearDown(self) -> None:
        for payload in _bash_jobs.listing():
            if payload["state"] == "running":
                _bash_jobs.stop(payload["job_key"])
        _bash_jobs.JOBS_ROOT = self._orig_root
        if self._orig_state is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_state
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._state, ignore_errors=True)

    def _await_current(self, timeout: float = 10.0) -> dict:
        """The run the server says it is waiting on, once it says so."""
        import json
        path = os.path.join(_bash_divert._dir(), "current.json")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with open(path) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                time.sleep(0.02)
        self.fail("the server never announced a foreground run")

    def test_a_timeout_returns_the_output_produced_before_the_kill(self) -> None:
        # The headline property. Before this, a timeout killed the process group AND
        # threw away everything it had written, so the whole wait bought nothing and
        # the model re-ran from scratch. The kill stays — it is what bounds a command
        # gone astray — but the bytes now come back with it.
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
        self.assertEqual(os.listdir(_bash_jobs.JOBS_ROOT), [])

    def test_a_completed_blocking_run_leaves_no_job_directory(self) -> None:
        # A job dir per `ls` would grow the cache without anyone holding a handle.
        server_bash.bash_run("echo hi")
        self.assertEqual(os.listdir(_bash_jobs.JOBS_ROOT), [])

    def test_a_refused_command_creates_no_job(self) -> None:
        # The blocking mirror of test_backgrounding_does_not_bypass_validation:
        # validation runs before anything is launched, on both paths.
        for command in ("echo hi & echo there", "bash -c 'echo hi'", "cat /etc/passwd"):
            with self.subTest(command=command):
                result = server_bash.bash_run(command, timeout=5)
                self.assertEqual(result["status"], "error")
        self.assertEqual(os.listdir(_bash_jobs.JOBS_ROOT), [])

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
            with open(os.path.join(_bash_divert._dir(), "divert"), "w") as fh:
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
            with open(os.path.join(_bash_divert._dir(), "divert"), "w") as fh:
                fh.write(current["job_key"])

        threading.Thread(target=divert, daemon=True).start()
        result = server_bash.bash_run("sleep 4", timeout=60)
        self.assertIn(result["job_key"], os.listdir(_bash_jobs.JOBS_ROOT))

    def test_a_divert_naming_another_run_is_not_consumed(self) -> None:
        # Two clients share one state dir. A request must name the run it meant.
        def divert() -> None:
            self._await_current()
            with open(os.path.join(_bash_divert._dir(), "divert"), "w") as fh:
                fh.write("20200101T000000Z-dead")

        threading.Thread(target=divert, daemon=True).start()
        result = server_bash.bash_run("echo hi; sleep 1", timeout=30)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("background_job", result)
