"""File tools take absolute paths only — the structural fix for misplacement.

Two runs were asked to create a solver *outside* the `codes` directory. Both wrote
it inside, and both reported the constraint satisfied. The cause was not a missing
fact: `write_file` accepted "absolute or relative to server start directory", and a
relative path was silently resolved against a root the model had to infer. Two
prompt-level attempts to make that inference reliable failed.

Requiring an absolute path removes the inference. There is no resolution step left
to get wrong, and the destination is stated in the call itself.

Pinned here: the rejection covers every mutating tool including the batch path, the
error is *self-correcting* (it names the path the relative form would have produced),
nothing is written on rejection, and the new precondition does not weaken the
sandbox or catch internal helpers.

Pure-Python + temp dirs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "workspace"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_files as sf  # noqa: E402


class _WorkspaceFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._old_root = sf.ROOT_DIR_ABS
        sf.ROOT_DIR_ABS = self.root

    def tearDown(self) -> None:
        sf.ROOT_DIR_ABS = self._old_root
        self._tmp.cleanup()

    def abs_path(self, name: str = "t.py") -> str:
        return os.path.join(self.root, name)

    def seed(self, name: str = "t.py", text: str = "alpha = 1\n") -> str:
        p = self.abs_path(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
        return p


class RelativePathsAreRejectedTests(_WorkspaceFixture):
    def test_every_mutating_tool_rejects_a_relative_path(self):
        self.seed()
        calls = {
            "write_file": lambda: sf.write_file(path="t.py", content="x = 1\n", overwrite=True),
            "append_file": lambda: sf.append_file(path="t.py", content="x = 1\n"),
            "delete_file": lambda: sf.delete_file(path="t.py", confirm=True),
            "replace_in_file": lambda: sf.replace_in_file(
                path="t.py", old_text="alpha = 1", new_text="beta = 2"),
            "replace_lines": lambda: sf.replace_lines(
                path="t.py", start_line=1, end_line=1, new_content="beta = 2\n"),
            "replace_all_in_file": lambda: sf.replace_all_in_file(
                path="t.py", old_text="alpha = 1", new_text="beta = 2", confirm=True),
        }
        for name, call in calls.items():
            res = call()
            self.assertEqual(res.get("status"), "error", f"{name}: {res}")
            self.assertIn("absolute path", res.get("error", "").lower(), name)

    def test_rejection_names_the_workspace_resolved_candidate(self):
        # The property that makes the rejection a one-step correction instead of an
        # obstacle: the model is handed the path it almost certainly meant, and the
        # root is stated at the moment placement is being decided.
        res = sf.write_file(path="wave_solver_2d/solver.py", content="x = 1\n")
        self.assertEqual(res.get("status"), "error")
        blob = res.get("error", "") + res.get("hint", "")
        self.assertIn(os.path.join(self.root, "wave_solver_2d/solver.py"), blob)
        self.assertIn(self.root, blob)

    def test_nothing_is_written_when_rejected(self):
        res = sf.write_file(path="new_dir/solver.py", content="x = 1\n")
        self.assertEqual(res.get("status"), "error")
        self.assertFalse(os.path.exists(os.path.join(self.root, "new_dir")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "new_dir/solver.py")))

    def test_bare_filename_is_rejected(self):
        # The exact shape of the original failure: a bare name reads as "somewhere
        # neutral" but resolves into the very directory the task excluded.
        res = sf.write_file(path="solver.py", content="x = 1\n")
        self.assertEqual(res.get("status"), "error")
        self.assertFalse(os.path.exists(os.path.join(self.root, "solver.py")))

    def test_dot_relative_and_parent_relative_are_rejected(self):
        for rel in ("./solver.py", "../solver.py", "sub/../solver.py"):
            res = sf.write_file(path=rel, content="x = 1\n")
            self.assertEqual(res.get("status"), "error", rel)

    def test_empty_path_is_rejected(self):
        res = sf.write_file(path="", content="x = 1\n")
        self.assertEqual(res.get("status"), "error")

    def test_apply_edits_rejects_a_relative_path_inside_the_batch(self):
        # A batch must not be a way around the rule.
        self.seed(text="alpha = 1\n")
        edits = json.dumps([
            {"operation": "replace_in_file", "path": "t.py",
             "old_text": "alpha = 1", "new_text": "beta = 2"},
        ])
        res = sf.apply_edits(edits_json=edits, confirm=True)
        self.assertEqual(res.get("status"), "error")
        self.assertIn("absolute path", res.get("error", "").lower())
        self.assertIn("Edit #0", res.get("error", ""))
        # Rejected in the validation phase — the file is untouched.
        with open(self.abs_path()) as fh:
            self.assertEqual(fh.read(), "alpha = 1\n")

    def test_apply_edits_rejects_when_only_one_sub_edit_is_relative(self):
        self.seed(text="alpha = 1\nN = 10\n")
        edits = json.dumps([
            {"operation": "replace_in_file", "path": self.abs_path(),
             "old_text": "alpha = 1", "new_text": "beta = 2"},
            {"operation": "replace_in_file", "path": "t.py",
             "old_text": "N = 10", "new_text": "N = 20"},
        ])
        res = sf.apply_edits(edits_json=edits, confirm=True)
        self.assertEqual(res.get("status"), "error")
        self.assertIn("Edit #1", res.get("error", ""))
        with open(self.abs_path()) as fh:
            self.assertEqual(fh.read(), "alpha = 1\nN = 10\n")


class AbsolutePathsStillWorkTests(_WorkspaceFixture):
    def test_write_append_and_replace_round_trip(self):
        fp = self.abs_path()
        self.assertEqual(sf.write_file(path=fp, content="alpha = 1\n").get("status"), "ok")
        self.assertEqual(sf.append_file(path=fp, content="N = 10\n").get("status"), "ok")
        self.assertEqual(
            sf.replace_in_file(path=fp, old_text="alpha = 1", new_text="beta = 2").get("status"),
            "ok")
        self.assertEqual(
            sf.replace_lines(path=fp, start_line=2, end_line=2,
                             new_content="N = 20\n").get("status"), "ok")
        with open(fp) as fh:
            self.assertEqual(fh.read(), "beta = 2\nN = 20\n")

    def test_replace_all_and_delete(self):
        fp = self.seed(text="a = 1\na = 1\n")
        self.assertEqual(
            sf.replace_all_in_file(path=fp, old_text="a = 1", new_text="b = 2",
                                   confirm=True).get("status"), "ok")
        self.assertEqual(sf.delete_file(path=fp, confirm=True).get("status"), "ok")
        self.assertFalse(os.path.exists(fp))

    def test_apply_edits_with_absolute_paths(self):
        fp = self.seed(text="alpha = 1\n")
        edits = json.dumps([
            {"operation": "replace_in_file", "path": fp,
             "old_text": "alpha = 1", "new_text": "beta = 2"},
        ])
        self.assertEqual(sf.apply_edits(edits_json=edits, confirm=True).get("status"), "ok")
        with open(fp) as fh:
            self.assertEqual(fh.read(), "beta = 2\n")

    def test_creating_a_nested_directory_works(self):
        fp = os.path.join(self.root, "pkg", "sub", "mod.py")
        self.assertEqual(sf.write_file(path=fp, content="x = 1\n").get("status"), "ok")
        self.assertTrue(os.path.isfile(fp))


class PreconditionDoesNotWeakenTheSandboxTests(_WorkspaceFixture):
    def test_absolute_path_outside_the_workspace_is_still_refused(self):
        # The new check runs *before* _safe; it must not become a way past it.
        outside = os.path.join(tempfile.mkdtemp(), "evil.py")
        res = sf.write_file(path=outside, content="x = 1\n")
        self.assertEqual(res.get("status"), "error")
        self.assertFalse(os.path.exists(outside))

    def test_scratchpad_absolute_path_still_succeeds(self):
        # The standing root must survive the new precondition. MIMIR_SCRATCH_DIR is
        # pointed at a temp dir so the hermetic suite never writes into the real
        # scratchpad under /tmp.
        from state_paths import scratch_dir
        state = tempfile.mkdtemp()
        scratch = tempfile.mkdtemp()
        for d in (state, scratch):
            self.addCleanup(lambda p=d: __import__("shutil").rmtree(p, ignore_errors=True))
        with unittest.mock.patch.dict(
            os.environ, {"MIMIR_STATE_DIR": state, "MIMIR_SCRATCH_DIR": scratch}
        ):
            fp = os.path.join(scratch_dir(), "probe.py")
            res = sf.write_file(path=fp, content="x = 1\n")
            self.assertEqual(res.get("status"), "ok", res)
            self.assertTrue(os.path.isfile(fp))


class InternalHelpersAreUnaffectedTests(_WorkspaceFixture):
    def test_list_files_still_takes_a_relative_subdir(self):
        # Not a tool — it backs the files://list resource and legitimately passes
        # relative paths, which is why the check lives at the tool boundary rather
        # than inside _safe.
        self.seed()
        for subdir in ("", ".", "pkg"):
            os.makedirs(os.path.join(self.root, "pkg"), exist_ok=True)
            res = sf.list_files(subdir)
            self.assertEqual(res.get("status"), "ok", f"{subdir!r} -> {res}")
        # Unchanged: the sandbox still governs where it may look.
        self.assertEqual(sf.list_files("/").get("status"), "error")


class PathRoundTripTests(unittest.TestCase):
    """Every path the model *reads* must be usable where it *writes*.

    The invariant that makes the absolute-path rule coherent: the model copies
    paths, it never constructs them. If discovery hands back workspace-relative
    paths while file tools require absolute ones, the model has to join the root
    itself — which is precisely the inference the rule removed, reintroduced at the
    moment it is most likely to copy without thinking.

    Regression guard: the discovery pin literally says "use these paths directly",
    and for a while it said that above paths the next call would reject.
    """

    def test_pin_renders_absolute_paths(self):
        from mimir.client.config.constants import WORKSPACE_ROOT
        from mimir.client.prompt.system_prompt import build_discovery_pin_block
        pin = build_discovery_pin_block({
            "read_files": {"pkg/mod.py"},
            "dirty_written_files": {"out/solver.py"},
            "existing_paths": {"README.md"},
            "planned_edit_targets": {"pkg/next.py"},
            "prev_query_written_files": {"pkg/old.py"},
        })
        for rel in ("pkg/mod.py", "out/solver.py", "README.md",
                    "pkg/next.py", "pkg/old.py"):
            self.assertIn(os.path.join(WORKSPACE_ROOT, rel), pin, rel)
        # No bare relative form left for the model to copy.
        for line in pin.splitlines():
            stripped = line.strip()
            if stripped.endswith(".py") or stripped.endswith(".md"):
                self.assertTrue(os.path.isabs(stripped), line)

    def test_pin_display_does_not_change_stored_form(self):
        # Display-only: the gates compare workspace-relative paths and must not shift.
        from mimir.client.prompt.system_prompt import build_discovery_pin_block
        ec = {"read_files": {"pkg/mod.py"}, "dirty_written_files": {"out/solver.py"}}
        build_discovery_pin_block(ec)
        self.assertEqual(ec["read_files"], {"pkg/mod.py"})
        self.assertEqual(ec["dirty_written_files"], {"out/solver.py"})

    def test_search_returns_absolute_paths(self):
        import server_search as ss
        old = ss.SEARCH_ROOT
        with tempfile.TemporaryDirectory() as d:
            ss.SEARCH_ROOT = d
            try:
                os.makedirs(os.path.join(d, "pkg"))
                with open(os.path.join(d, "pkg", "mod.py"), "w") as fh:
                    fh.write("x = 1\n")

                r = ss.read_file_lines("pkg/mod.py")
                self.assertTrue(os.path.isabs(r["path"]), r)

                r = ss.list_directory("pkg")
                self.assertTrue(os.path.isabs(r["path"]), r)
                for e in r["entries"]:
                    self.assertTrue(os.path.isabs(e["path"]), e)

                r = ss.tree_summary("pkg", use_cache=False)
                self.assertTrue(os.path.isabs(r["path"]), r)
                # The tree's own root line, not just the echoed argument.
                self.assertIn(os.path.abspath(os.path.join(d, "pkg")), r["tree"])

                r = ss.read_files(["pkg/mod.py"])
                self.assertTrue(os.path.isabs(r["files"][0]["path"]), r)
            finally:
                ss.SEARCH_ROOT = old

    def test_a_searched_path_is_directly_writable(self):
        # The end-to-end property: copy what discovery returned into a file tool.
        import server_search as ss
        old_root = ss.SEARCH_ROOT
        with tempfile.TemporaryDirectory() as d:
            ss.SEARCH_ROOT = d
            old_files_root = sf.ROOT_DIR_ABS
            sf.ROOT_DIR_ABS = d
            try:
                with open(os.path.join(d, "mod.py"), "w") as fh:
                    fh.write("alpha = 1\n")
                found = ss.read_file_lines("mod.py")["path"]
                res = sf.replace_in_file(path=found, old_text="alpha = 1",
                                         new_text="beta = 2")
                self.assertEqual(res.get("status"), "ok", res)
            finally:
                ss.SEARCH_ROOT = old_root
                sf.ROOT_DIR_ABS = old_files_root


class DocstringsAdvertiseTheRuleTests(unittest.TestCase):
    def test_tool_docstrings_say_absolute_is_required(self):
        # The docstring is the model's only spec for the argument. The old wording
        # ("absolute or relative to server start directory") named a directory the
        # model had no way to learn — the bug in one sentence.
        for fn in (sf.write_file, sf.append_file, sf.replace_in_file,
                   sf.replace_lines, sf.replace_all_in_file):
            doc = fn.__doc__ or ""
            self.assertNotIn("relative to server start directory", doc, fn.__name__)
        self.assertIn("ABSOLUTE path", sf.write_file.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
