"""Detached ``bash_run(background=True)`` jobs: launch, terminal state, stop.

The blocking path is capped at 300s, which no real build fits inside. These cover the
part that makes a detached run usable rather than merely started: that a terminal state
is knowable afterwards (the exit code the shell records on its way out), that stopping
one takes the whole process group with it, and — the property that must not regress —
that detaching is a parameter on an already-validated command, not a way around the
validation. See ``_bash_jobs`` and ``server_bash.bash_run``.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
