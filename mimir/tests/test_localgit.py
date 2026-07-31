"""Tests for the local-git server (servers/workspace/server_localgit.py).

Covers the safety boundary (refused options, non-allowlisted subcommands) and the
read-only happy path against whatever git repository the tests run inside.

Run:
    python -m unittest tests.test_localgit -v
"""

import subprocess
import sys
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import server_localgit as lg  # noqa: E402


def _in_git_repo() -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=lg._WORKSPACE_ROOT, capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


class LocalGitBoundaryTests(unittest.TestCase):
    """Boundary checks — do not require a git repo (validation runs before exec)."""

    def test_refuses_dash_c_option(self) -> None:
        payload = lg._git(["-c", "alias.x=!sh", "status"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("Refused git option", payload["error"])

    def test_refuses_exec_path_option(self) -> None:
        payload = lg._git(["--exec-path=/tmp", "status"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("Refused git option", payload["error"])

    def test_refuses_non_allowlisted_subcommand(self) -> None:
        for sub in ("push", "commit", "checkout", "reset"):
            payload = lg._git([sub])
            self.assertEqual(payload["status"], "error", sub)
            self.assertIn("not allowed", payload["error"], sub)

    def test_empty_args_is_error(self) -> None:
        payload = lg._git([])
        self.assertEqual(payload["status"], "error")

    def test_git_show_requires_ref(self) -> None:
        payload = lg.git("show", ref="")
        self.assertEqual(payload["status"], "error")

    def test_git_blame_requires_path(self) -> None:
        payload = lg.git("blame", path="")
        self.assertEqual(payload["status"], "error")

    def test_unknown_op_is_structured_error(self) -> None:
        payload = lg.git("push")
        self.assertEqual(payload["status"], "error")
        self.assertIn("Unknown git op", payload["error"])


@unittest.skipUnless(_in_git_repo(), "not inside a git repository")
class LocalGitReadOnlyTests(unittest.TestCase):
    """Happy-path read-only checks against the surrounding repository."""

    def test_git_status_ok_and_structured(self) -> None:
        payload = lg.git("status")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("stdout", payload)
        self.assertEqual(payload["subcommand"], "status")
        self.assertEqual(payload["cwd"], lg._WORKSPACE_ROOT)

    def test_git_log_respects_max_count(self) -> None:
        payload = lg.git("log", max_count=2)
        self.assertEqual(payload["status"], "ok")
        nonempty = [ln for ln in payload["stdout"].splitlines() if ln.strip()]
        self.assertLessEqual(len(nonempty), 2)

    def test_git_branches_ok(self) -> None:
        payload = lg.git("branches")
        self.assertEqual(payload["status"], "ok")

    def test_git_diff_ok(self) -> None:
        payload = lg.git("diff")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["subcommand"], "diff")


if __name__ == "__main__":
    unittest.main()
