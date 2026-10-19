"""When this session's shell commands ran: the evidence that a job submitted by a launcher
is this session's own, and never another session's."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from mimir.shared import shell_activity
from mimir.shared.jobs import EXIT_CODE, META


class ShellActivityTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = tmp.name
        env = patch.dict(os.environ, {"MIMIR_STATE_DIR": self.state, "MIMIR_SESSION_ID": "s-a"})
        env.start()
        self.addCleanup(env.stop)

    def _record(self, session: str, start: float, end: float | None, job_dir: str = "") -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": session}):
            shell_activity.record(start, end, job_dir)

    def _job_dir(self, *, pid: int, exited: bool) -> str:
        job_dir = tempfile.mkdtemp(dir=self.state)
        with open(os.path.join(job_dir, META), "w") as fh:
            json.dump({"pid": pid}, fh)
        if exited:
            with open(os.path.join(job_dir, EXIT_CODE), "w") as fh:
                fh.write("0")
        return job_dir

    def test_a_submission_during_one_of_its_commands_is_its_own(self) -> None:
        self._record("s-a", 1000.0, 1100.0)
        self.assertTrue(shell_activity.submitted_by_this_session(1050.0))
        self.assertFalse(shell_activity.submitted_by_this_session(5000.0))

    def test_a_sub_agents_command_counts_for_its_parent(self) -> None:
        self._record("s-a/subagents/sub-1", 1000.0, 1100.0)
        self.assertTrue(shell_activity.submitted_by_this_session(1050.0))

    def test_another_session_at_the_same_moment_makes_it_nobodys(self) -> None:
        self._record("s-a", 1000.0, 1100.0)
        self._record("s-b", 1040.0, 1060.0)
        self.assertFalse(shell_activity.submitted_by_this_session(1050.0))

    def test_a_detached_run_stays_open_while_it_runs(self) -> None:
        start = time.time() - 600
        self._record("s-a", start, None, self._job_dir(pid=os.getpid(), exited=False))
        self.assertTrue(shell_activity.submitted_by_this_session(time.time() - 5))

    def test_a_detached_run_closes_when_it_exits(self) -> None:
        job_dir = self._job_dir(pid=os.getpid(), exited=True)
        os.utime(os.path.join(job_dir, EXIT_CODE), (1100.0, 1100.0))
        self._record("s-a", 1000.0, None, job_dir)
        self.assertTrue(shell_activity.submitted_by_this_session(1050.0))
        self.assertFalse(shell_activity.submitted_by_this_session(1100.0 + 3600))

    def test_a_session_that_ran_nothing_says_so(self) -> None:
        self.assertFalse(shell_activity.ran_commands())
        self._record("s-a", 1000.0, 1100.0)
        self.assertTrue(shell_activity.ran_commands())

    def test_every_bash_run_is_recorded(self) -> None:
        from mimir.servers.workspace import bash_exec
        before = time.time()
        bash_exec.run_blocking("true", self.state, 10)
        self.assertTrue(shell_activity.submitted_by_this_session(before + 0.5))


if __name__ == "__main__":
    unittest.main()
