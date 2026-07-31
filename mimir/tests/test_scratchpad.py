"""The agent scratchpad: a writable place outside the workspace.

Without one, the only writable location is the user's own tree, so every throwaway
probe script becomes indistinguishable from produced work — it lands in the repo
*and* in the change ledger, and then demands validation before the run can
conclude. The scratchpad is granted as a *standing* root (system-granted, never
prompted) and is excluded from deliverable accounting.

Composed from two pre-existing seams rather than new machinery: ``state_dir()``
and the ``extra_roots`` sandbox parameter, which already admitted absolute paths
outside the workspace.

Pure-Python + temp dirs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mimir.servers._shared.root_paths import resolve_path_in_root
from mimir.servers._shared.state_paths import (
    active_session_id,
    scratch_dir,
    standing_roots,
)


class ScratchDirResolutionTests(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.base, ignore_errors=True))

    def test_falls_back_outside_a_session(self):
        self.assertEqual(scratch_dir(self.base), os.path.join(self.base, "scratch"))

    def test_is_session_scoped_when_a_session_is_active(self):
        with open(os.path.join(self.base, "active_session"), "w") as fh:
            fh.write("sess-42\n")
        self.assertEqual(active_session_id(self.base), "sess-42")
        self.assertEqual(
            scratch_dir(self.base),
            os.path.join(self.base, "sessions", "sess-42", "scratch"),
        )

    def test_unreadable_sidecar_means_no_session(self):
        # Best-effort by design: a missing/corrupt pointer degrades to the shared
        # scratchpad rather than raising inside a sandbox check.
        self.assertEqual(active_session_id(self.base), "")
        self.assertEqual(scratch_dir(self.base), os.path.join(self.base, "scratch"))

    def test_both_session_and_shared_roots_are_granted(self):
        # A session switch mid-run must not revoke a path already being written.
        with open(os.path.join(self.base, "active_session"), "w") as fh:
            fh.write("s1")
        roots = standing_roots(self.base)
        self.assertIn(os.path.join(self.base, "sessions", "s1", "scratch"), roots)
        self.assertIn(os.path.join(self.base, "scratch"), roots)

    def test_resolution_does_not_create_directories(self):
        # A sandbox check runs on every call; it must not materialise state.
        scratch_dir(self.base)
        standing_roots(self.base)
        self.assertFalse(os.path.exists(os.path.join(self.base, "scratch")))


class ScratchSandboxTests(unittest.TestCase):
    """The sandbox admits the scratchpad and nothing else new."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.state = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.state, ignore_errors=True))
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
        self.addCleanup(lambda: __import__("shutil").rmtree(self.state, ignore_errors=True))
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
