"""Annotations appended to a tool result: the fork hint and the repeat-read pointer.

Both live in ``tool_execution.executor`` next to READ_HINT / MORE_CONTENT /
VERDICT_DUE, and both say their piece at the call they concern rather than at the end
of the turn — which is the only moment the cheap repair is still one call away.

Pure-Python + stubs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import json
import os
import tempfile
import types
import unittest

from mimir.client.context.capabilities import EDIT, READ, CACHEABLE, ToolCaps
from mimir.client.context.execution_context import build_execution_context
from mimir.client.tool_execution import executor as ex


def _agent():
    caps = {
        "write_file": ToolCaps(name="write_file", capabilities=frozenset({EDIT})),
        "read_file_lines": ToolCaps(
            name="read_file_lines", capabilities=frozenset({READ, CACHEABLE})
        ),
    }
    return types.SimpleNamespace(
        _normalize_workspace_path=lambda p: p or "",
        tool_caps=caps,
    )


def _created(path: str) -> str:
    return json.dumps({"status": "ok", "operation": "created", "path": path})


class ForkHintTests(unittest.TestCase):
    """A second copy of a file already written this query is worth one remark."""

    def _hint(self, ec, path, result=None):
        # record_tool_observation has already run by the time the hint is built, so
        # the file just written is itself in the dirty set: the check must not read
        # it as its own fork.
        ec["dirty_written_files"].add(path)
        return ex._build_fork_hint(
            _agent(), "write_file", {"path": path},
            result if result is not None else _created(path), ec,
        )

    def test_suffixed_stem_is_read_as_a_fork(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/convergence_test.py")
        hint = self._hint(ec, "work/convergence_test_fixed.py")
        self.assertIn("FORK_SUSPECTED", hint)
        self.assertIn("convergence_test.py", hint)

    def test_short_stem_suffix_still_matches(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/solver.py")
        self.assertIn("FORK_SUSPECTED", self._hint(ec, "work/solver_ghost.py"))

    def test_unrelated_stem_is_left_alone(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/solver.py")
        self.assertEqual(self._hint(ec, "work/driver.py"), "")

    def test_a_sibling_in_another_directory_is_not_a_fork(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("other/solver.py")
        self.assertEqual(self._hint(ec, "work/solver_ghost.py"), "")

    def test_overwrite_of_an_existing_file_is_the_wanted_behaviour(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/convergence_test.py")
        updated = json.dumps({"status": "ok", "operation": "updated"})
        self.assertEqual(
            self._hint(ec, "work/convergence_test_fixed.py", updated), ""
        )

    def test_probe_name_outside_a_tests_dir_is_flagged(self) -> None:
        ec = build_execution_context()
        hint = self._hint(ec, "work/test_laplacian.py")
        self.assertIn("PROBE_PLACEMENT", hint)

    def test_the_same_name_inside_a_tests_dir_is_where_it_belongs(self) -> None:
        ec = build_execution_context()
        self.assertEqual(self._hint(ec, "pkg/tests/test_laplacian.py"), "")

    def test_it_speaks_once_per_query(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/solver.py")
        self.assertIn("FORK_SUSPECTED", self._hint(ec, "work/solver_ghost.py"))
        ec["dirty_written_files"].add("work/driver.py")
        self.assertEqual(self._hint(ec, "work/driver_v2.py"), "")

    def test_the_scratchpad_is_exactly_where_these_files_belong(self) -> None:
        from mimir.servers._shared.state_paths import scratch_dir
        from mimir.client.config.constants import STATE_DIR
        scratch = scratch_dir(STATE_DIR)
        ec = build_execution_context()
        ec["dirty_written_files"].add(os.path.join(scratch, "probe.py"))
        self.assertEqual(
            self._hint(ec, os.path.join(scratch, "probe_fixed.py")), ""
        )


class RepeatReadTests(unittest.TestCase):
    """A read the model already holds is answered with a pointer, not a second copy."""

    def _ack(self, ec, path):
        return ex._repeat_read_acknowledgement(
            _agent(), "read_file_lines", {"path": path}, ec
        )

    def test_a_held_file_gets_a_pointer_instead_of_its_content(self) -> None:
        ec = build_execution_context()
        ec["read_files"].add("solver.py")
        ack = self._ack(ec, "solver.py")
        self.assertIn("solver.py", ack)
        self.assertIn("not repeated here", ack)
        self.assertEqual(json.loads(ack)["status"], "ok")

    def test_a_file_evicted_to_force_a_reread_is_served_in_full(self) -> None:
        # What _observe_edit_outcome does after two failed edits: drop the path so the
        # discovery gate demands a fresh read. Answering "you already have it" there
        # would defeat the mechanism asking for it.
        ec = build_execution_context()
        self.assertEqual(self._ack(ec, "solver.py"), "")

    def test_a_truncated_history_retracts_the_claim(self) -> None:
        ec = build_execution_context()
        ec["read_files"].add("solver.py")
        ec["history_truncated"] = True
        self.assertEqual(self._ack(ec, "solver.py"), "")

    def test_a_pathless_call_has_no_file_to_reason_about(self) -> None:
        ec = build_execution_context()
        self.assertEqual(self._ack(ec, ""), "")


class PathStampTests(unittest.TestCase):
    """The stamp is what makes answering from the cache safe."""

    def test_a_write_outside_the_edit_tools_changes_the_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "solver.py")
            with open(path, "w") as fh:
                fh.write("x = 1\n")
            first = ex._path_stamp(_agent(), {"path": path})
            self.assertIsNotNone(first)
            # A shell redirect or the user's own editor: no tool call to notice it.
            with open(path, "w") as fh:
                fh.write("x = 1\ny = 2\n")
            self.assertNotEqual(ex._path_stamp(_agent(), {"path": path}), first)

    def test_an_unreadable_or_absent_path_stamps_to_nothing(self) -> None:
        self.assertIsNone(ex._path_stamp(_agent(), {"path": "/nope/missing.py"}))
        self.assertIsNone(ex._path_stamp(_agent(), {}))


class ForkBaseStemTests(unittest.TestCase):
    def test_it_strips_one_trailing_word(self) -> None:
        self.assertEqual(ex._fork_base_stem("convergence_test_fixed"), "convergence_test")
        self.assertEqual(ex._fork_base_stem("solver_ghost"), "solver")

    def test_a_stem_with_no_suffix_yields_nothing(self) -> None:
        self.assertEqual(ex._fork_base_stem("solver"), "")

    def test_a_remainder_too_short_to_mean_anything_is_discarded(self) -> None:
        self.assertEqual(ex._fork_base_stem("a_b"), "")


if __name__ == "__main__":
    unittest.main()
