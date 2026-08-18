"""The agent scratchpad: a writable place outside the workspace, under the temp dir.

Without one, the only writable location is the user's own tree, so every throwaway
probe script becomes indistinguishable from produced work — it lands in the repo
*and* in the change ledger, and then demands validation before the run can
conclude. The scratchpad is granted as a *standing* root (system-granted, never
prompted) and is excluded from deliverable accounting.

It lives under ``<TMPDIR or /tmp>/mimir-<uid>-<workspace-id>`` — where throwaway work
belongs, and where the OS reclaims it. That location is world-writable, so the
ownership vetting in ``ensure_scratch_home`` is part of the contract, not a detail:
an existing path at our name may be someone else's symlink.

Pure-Python + temp dirs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from unittest.mock import patch

from mimir.servers._shared.root_paths import resolve_path_in_root
from mimir.servers._shared.state_paths import (
    active_session_id,
    ensure_scratch_home,
    scratch_dir,
    scratch_home,
    standing_roots,
)


class ScratchHomeResolutionTests(unittest.TestCase):
    """Where the home lands, and that resolving it never touches the disk."""

    def test_env_override_wins(self):
        with patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": "/somewhere/else"}):
            self.assertEqual(scratch_home(), "/somewhere/else")

    def test_default_is_under_the_temp_dir_and_scoped_by_uid_and_workspace(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        with patch.dict(
            os.environ, {"TMPDIR": tmp, "MCP_FILES_ROOT": "/w/proj"}, clear=False
        ):
            os.environ.pop("MIMIR_SCRATCH_DIR", None)
            home = scratch_home()
        self.assertEqual(os.path.dirname(home), tmp)
        # Two users, and two checkouts sharing a basename, must not collide.
        self.assertTrue(os.path.basename(home).startswith(f"mimir-{os.getuid()}-proj-"))

    def test_resolution_does_not_create_directories(self):
        # A sandbox check runs on every call; it must not materialise state.
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        target = os.path.join(tmp, "home")
        with patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": target}):
            scratch_dir()
            standing_roots()
        self.assertFalse(os.path.exists(target))


class ScratchDirResolutionTests(unittest.TestCase):
    """The per-session subdirectory, and what the standing grant covers."""

    def setUp(self):
        self.base = tempfile.mkdtemp()       # a fake state dir (session sidecar only)
        self.home = tempfile.mkdtemp()       # a fake scratchpad home
        for d in (self.base, self.home):
            self.addCleanup(lambda p=d: __import__("shutil").rmtree(p, ignore_errors=True))
        patcher = patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": self.home})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _activate_session(self, sid: str) -> None:
        with open(os.path.join(self.base, "active_session"), "w") as fh:
            fh.write(sid + "\n")

    def test_falls_back_to_the_home_outside_a_session(self):
        self.assertEqual(scratch_dir(self.base), self.home)

    def test_is_session_scoped_when_a_session_is_active(self):
        self._activate_session("sess-42")
        self.assertEqual(active_session_id(self.base), "sess-42")
        self.assertEqual(scratch_dir(self.base), os.path.join(self.home, "sess-42"))

    def test_unreadable_sidecar_means_no_session(self):
        # Best-effort by design: a missing/corrupt pointer degrades to the shared
        # scratchpad rather than raising inside a sandbox check.
        self.assertEqual(active_session_id(self.base), "")
        self.assertEqual(scratch_dir(self.base), self.home)

    def test_the_grant_is_the_home_so_a_session_switch_cannot_revoke_a_path(self):
        # One root covering both: a session switch mid-run must not revoke a path
        # already being written.
        self._activate_session("s1")
        self.assertEqual(standing_roots(self.base), [self.home])
        self._activate_session("s2")
        self.assertEqual(standing_roots(self.base), [self.home])


class EnsureScratchHomeTests(unittest.TestCase):
    """The one function that touches the disk. /tmp is world-writable."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.home = os.path.join(self.tmp, "mimir-home")

    def _ensure(self) -> str:
        with patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": self.home}):
            return ensure_scratch_home()

    def test_creates_it_private(self):
        self.assertEqual(self._ensure(), self.home)
        self.assertEqual(stat.S_IMODE(os.stat(self.home).st_mode), 0o700)

    def test_is_idempotent(self):
        self._ensure()
        self.assertEqual(self._ensure(), self.home)

    def test_tightens_loose_permissions(self):
        os.makedirs(self.home, mode=0o777)
        os.chmod(self.home, 0o777)
        self.assertEqual(self._ensure(), self.home)
        self.assertEqual(stat.S_IMODE(os.stat(self.home).st_mode), 0o700)

    def test_refuses_a_symlink(self):
        # Someone else choosing where our writes land is exactly the /tmp hazard.
        target = os.path.join(self.tmp, "target")
        os.makedirs(target)
        os.symlink(target, self.home)
        self.assertEqual(self._ensure(), "")

    def test_refuses_a_non_directory(self):
        with open(self.home, "w") as fh:
            fh.write("not a dir")
        self.assertEqual(self._ensure(), "")

    def test_refuses_a_foreign_owner(self):
        os.makedirs(self.home, mode=0o700)
        with patch("mimir.servers._shared.state_paths.os.getuid", return_value=-1):
            self.assertEqual(self._ensure(), "")

    def test_refuses_when_it_cannot_create(self):
        # An unwritable parent is a refusal, not a traceback out of client startup.
        parent = os.path.join(self.tmp, "locked")
        os.makedirs(parent, mode=0o500)
        self.addCleanup(os.chmod, parent, 0o700)
        self.home = os.path.join(parent, "home")
        self.assertEqual(self._ensure(), "")


class ScratchSandboxTests(unittest.TestCase):
    """The sandbox admits the scratchpad and nothing else new."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.state = tempfile.mkdtemp()
        home = tempfile.mkdtemp()
        for d in (self.root, self.state, home):
            self.addCleanup(lambda p=d: __import__("shutil").rmtree(p, ignore_errors=True))
        patcher = patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": home})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scratch = scratch_dir(self.state)

    def _resolve(self, path):
        return resolve_path_in_root(
            path, self.root, "file root", extra_roots=standing_roots(self.state),
        )

    def test_scratch_path_is_admitted(self):
        target = os.path.join(self.scratch, "probe.py")
        self.assertEqual(self._resolve(target), os.path.abspath(target))

    def test_workspace_path_still_admitted(self):
        target = os.path.join(self.root, "src.py")
        self.assertEqual(self._resolve(target), os.path.abspath(target))

    def test_arbitrary_outside_path_still_refused(self):
        # The grant must be the scratchpad specifically, not "outside is fine now".
        with self.assertRaises(ValueError):
            self._resolve("/etc/passwd")

    def test_sibling_of_scratch_is_refused(self):
        # Prefix matching must be path-segment aware: "<scratch>_evil" is not inside.
        with self.assertRaises(ValueError):
            self._resolve(self.scratch + "_evil/x.py")

    def test_relative_paths_still_resolve_into_the_workspace(self):
        # Unchanged: a bare filename is workspace-relative, never scratch-relative.
        self.assertEqual(self._resolve("f.py"), os.path.join(self.root, "f.py"))


class ScratchIsNotADeliverableTests(unittest.TestCase):
    """Scratch writes must not enter the change ledger or demand validation."""

    def setUp(self):
        self.state = tempfile.mkdtemp()
        home = tempfile.mkdtemp()
        for d in (self.state, home):
            self.addCleanup(lambda p=d: __import__("shutil").rmtree(p, ignore_errors=True))
        env = patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": home})
        env.start()
        self.addCleanup(env.stop)
        self.scratch = scratch_dir(self.state)
        patcher = patch(
            "mimir.client.tool_execution.validation.scratch_roots",
            return_value=[os.path.realpath(self.scratch)],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_is_scratch_path_discriminates(self):
        from mimir.client.tool_execution.validation import is_scratch_path
        self.assertTrue(is_scratch_path(os.path.join(self.scratch, "probe.py")))
        self.assertFalse(is_scratch_path("/some/workspace/solver.py"))
        self.assertFalse(is_scratch_path(""))

    def test_scratch_write_is_not_recorded_as_produced_work(self):
        from mimir.client.context.execution_context import build_execution_context
        from mimir.client.guardrails.observations import _record_code_edit
        ec = build_execution_context()
        _record_code_edit(ec, os.path.join(self.scratch, "probe.py"))
        self.assertFalse(ec["dirty_written_files"])
        self.assertFalse(ec["code_mutation_started"])

    def test_workspace_write_is_still_recorded(self):
        from mimir.client.context.execution_context import build_execution_context
        from mimir.client.guardrails.observations import _record_code_edit
        ec = build_execution_context()
        _record_code_edit(ec, "solver.py")
        self.assertIn("solver.py", ec["dirty_written_files"])
        self.assertTrue(ec["code_mutation_started"])

    def test_scratch_work_produces_no_ledger(self):
        from mimir.client.context.execution_context import build_execution_context
        from mimir.client.guardrails.observations import _record_code_edit
        from mimir.client.query_engine.finalize import _annotate_answer_with_changes
        ec = build_execution_context()
        _record_code_edit(ec, os.path.join(self.scratch, "probe.py"))
        self.assertEqual(_annotate_answer_with_changes("Done.", ec), "Done.")


if __name__ == "__main__":
    unittest.main()
