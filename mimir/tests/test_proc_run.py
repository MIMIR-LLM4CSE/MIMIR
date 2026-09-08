"""A timeout must stop the work, not just stop waiting for it.

`subprocess.run(..., timeout=...)` kills the process it launched and nothing that
process forked. The bash tool runs `bash -c "<preamble><command>"`, which is compound,
so bash forks rather than execs — the kill landed on bash and the fork ran on. Observed:
`find / -name _proxy_runner.py` outliving its 30s timeout by an hour and forty minutes,
three at a time, while a ratchet next door was timing a solver on the same disk.

These tests use a marker file a grandchild keeps writing: if it is still growing after
the timeout returned, the work was never stopped.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "servers", "_shared"))

import proc_run  # noqa: E402


def _forking_script(marker: str) -> str:
    """A shell script whose *child* outlives the shell: the shape bash -c produces."""
    return (
        f"(while true; do echo tick >> {marker!r}; sleep 0.05; done) & "
        "sleep 30"
    )


class TimeoutKillsTheTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.marker = tempfile.mkstemp(prefix="proc_run_", suffix=".marker")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.marker) and os.remove(self.marker))

    def _size(self) -> int:
        return os.path.getsize(self.marker)

    def test_a_forked_grandchild_does_not_outlive_the_timeout(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired):
            proc_run.run(
                ["bash", "--noprofile", "--norc", "-c", _forking_script(self.marker)],
                timeout=0.6, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        settled = self._size()
        time.sleep(0.5)  # comfortably several of the grandchild's 0.05s ticks
        self.assertEqual(self._size(), settled,
                         "the grandchild kept writing after the timeout returned")

    def test_subprocess_run_is_what_this_replaces(self) -> None:
        """The same script through subprocess.run leaks — this is the defect, pinned.

        Kept so the reason for proc_run cannot be lost: if a future Python fixes this
        on its own, this test fails and the wrapper can go.
        """
        with self.assertRaises(subprocess.TimeoutExpired):
            subprocess.run(
                ["bash", "--noprofile", "--norc", "-c", _forking_script(self.marker)],
                timeout=0.6, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        settled = self._size()
        time.sleep(0.5)
        leaked = self._size() > settled
        if leaked:  # expected today; clean up the process we deliberately leaked
            subprocess.run(["pkill", "-f", self.marker],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertTrue(leaked, "subprocess.run no longer leaks — proc_run may be dropped")


class NormalOperationTests(unittest.TestCase):
    def test_it_returns_a_completed_process(self) -> None:
        out = proc_run.run(["echo", "hello"], timeout=10,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertEqual(out.stdout.strip(), "hello")
        self.assertIsInstance(out, subprocess.CompletedProcess)

    def test_a_non_zero_exit_is_returned_not_raised(self) -> None:
        out = proc_run.run(["bash", "-c", "exit 3"], timeout=10)
        self.assertEqual(out.returncode, 3)

    def test_the_child_is_never_in_the_callers_process_group(self) -> None:
        """The safety property: killing the group can never reach MIMIR itself."""
        out = proc_run.run(["bash", "-c", "echo $$; ps -o pgid= -p $$"], timeout=10,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        child_pgid = int(out.stdout.split()[-1])
        self.assertNotEqual(child_pgid, os.getpgid(0))

    def test_timeout_expired_carries_the_command(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            proc_run.run(["sleep", "5"], timeout=0.3)
        self.assertIn("sleep", str(caught.exception.cmd))


if __name__ == "__main__":
    unittest.main()
