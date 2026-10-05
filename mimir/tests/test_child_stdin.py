"""No command a tool runs can read the server's stdin.

A server's stdin is its MCP protocol pipe: the client's JSON-RPC requests arrive on it.
A child inherits it unless told otherwise, and plenty of ordinary commands read stdin —
``ssh`` without ``-n`` drains it, ``cat`` and ``head`` consume it outright. Such a command
eats the client's traffic: a request sent while it runs (a background-job watcher polling
``bash_job``, say) is swallowed and never answered, and a partial steal desynchronises the
framing so every later call on that session hangs or fails on a dead transport — while the
server itself looks perfectly healthy.

So every spawn in every server redirects stdin from ``/dev/null``. Checked here on the
real launcher with a real stdin-reading command, not by inspecting kwargs: the property is
that the bytes stay unread, whatever the layers in between do.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import proc_run      # noqa: E402
import server_bash   # noqa: E402


# A parent holding bytes on its own stdin, running a command through the layer under
# test, then reporting whether the child took any of them and what is left.
_PROBE = """
import os, sys
sys.path[:0] = [{servers!r} + "/_shared", {servers!r} + "/workspace"]
{run}
left = os.read(0, 64).decode()
print("LEFT:" + left)
"""


def _probe(run_snippet: str) -> str:
    """Run *run_snippet* in a child whose stdin holds a known marker; return what is left.

    The marker stands for a client request already in the pipe. Whatever the snippet
    launches must not be able to take it.
    """
    source = _PROBE.format(servers=str(SERVERS_DIR), run=run_snippet)
    proc = subprocess.run(
        [sys.executable, "-c", source],
        input=b"REQUEST-BYTES\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120,
    )
    out = proc.stdout.decode()
    assert "LEFT:" in out, f"probe failed: {proc.stderr.decode()[-2000:]}"
    # The probe's own print adds a newline of its own; the marker is what matters.
    return out.split("LEFT:", 1)[1].strip()


class ProcRunTests(unittest.TestCase):
    """``proc_run.run`` is what the servers use for a command with a deadline."""

    def test_a_child_cannot_read_the_parents_stdin(self) -> None:
        left = _probe(
            "import proc_run\n"
            "r = proc_run.run(['cat'], timeout=20, stdout=-1)\n"
            "print('CHILD-SAW:' + r.stdout.decode().strip())\n"
        )
        self.assertEqual(left, "REQUEST-BYTES", "the child ate the client's request")

    def test_an_explicit_stdin_is_still_honoured(self) -> None:
        """The default must not take the choice away from a caller that needs to feed one."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("fed-in")
            path = fh.name
        self.addCleanup(os.unlink, path)
        with open(path) as fed:
            proc = proc_run.run(["cat"], timeout=20, stdin=fed, stdout=subprocess.PIPE)
        self.assertEqual(proc.stdout, b"fed-in")


class BashJobTests(unittest.TestCase):
    """Every ``bash_run`` goes through the job launcher — blocking or detached."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-stdin-state-")
        self._orig = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._tmp

    def tearDown(self) -> None:
        if self._orig is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_a_blocking_command_cannot_read_the_servers_stdin(self) -> None:
        """The shape of the real failure: ``ssh host "srun …"`` draining the pipe."""
        left = _probe(
            f"os.environ['MIMIR_STATE_DIR'] = {self._tmp!r}\n"
            "import server_bash\n"
            "server_bash.bash_run('timeout 5 cat', timeout=20)\n"
        )
        self.assertEqual(left, "REQUEST-BYTES", "bash_run's child ate the request")

    def test_a_detached_job_cannot_read_it_either(self) -> None:
        """A job outlives the call, so an inherited pipe would be held for its whole run."""
        left = _probe(
            f"os.environ['MIMIR_STATE_DIR'] = {self._tmp!r}\n"
            "import time, server_bash\n"
            "server_bash.bash_run('timeout 5 cat', background=True)\n"
            "time.sleep(2)\n"
        )
        self.assertEqual(left, "REQUEST-BYTES", "a detached job ate the request")

    def test_a_command_reading_stdin_sees_eof_rather_than_hanging(self) -> None:
        """``/dev/null``, not a closed fd: a reader must end, not fail or block."""
        result = server_bash.bash_run("cat; echo rc=$?", timeout=30)
        self.assertEqual(result["status"], "ok", result)
        self.assertIn("rc=0", result["stdout"])


if __name__ == "__main__":
    unittest.main()
