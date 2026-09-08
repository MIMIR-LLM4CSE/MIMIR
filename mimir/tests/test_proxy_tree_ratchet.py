"""The ratchet optimizes the real code, and remembers whole tree states.

Two recorded sessions produced the same shape: asked to speed up a solver, the model
wrote a self-contained script that reproduced it — 189 lines mirroring a 307-line
package — and the ratchet optimized the copy. Nothing said not to. `proxy_manage`
asked for a runnable command and `proxy_eval(init)` answered "Modify '<the harness>'
between runs", so making "what I edit" and "what I optimize" coincide was the cheapest
way to satisfy both.

The cost is not the manual port afterwards: the ratchet's accuracy constraints then hold
for the duplicate and for nothing that ships. These tests pin the contract that makes
that shape inexpressible, and the atomicity that lets it generalize past one file.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

_PROXY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "servers", "proxy")
sys.path.insert(0, os.path.abspath(_PROXY))
sys.path.insert(0, os.path.abspath(os.path.join(_PROXY, "..", "_shared")))


class _Workspace(unittest.TestCase):
    """A workspace holding a harness plus a two-file package it imports."""

    def setUp(self) -> None:
        self.wt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.wt, True)
        os.environ["MCP_FILES_ROOT"] = self.wt
        self.addCleanup(os.environ.pop, "MCP_FILES_ROOT", None)
        os.makedirs(os.path.join(self.wt, "pkg"))
        self.harness = os.path.join(self.wt, "harness.py")
        self.a = os.path.join(self.wt, "pkg", "solver.py")
        self.b = os.path.join(self.wt, "pkg", "kernel.py")
        self._write(self.harness, "import pkg.solver\n")
        self._write(self.a, "A0\n")
        self._write(self.b, "B0\n")

    @staticmethod
    def _write(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _read(self, path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()


class HarnessContractTests(_Workspace):
    """init refuses the shape that produced the duplicate."""

    def _check(self, paths, src=None):
        from _ops import eval_session
        return eval_session._check_optimize_paths(paths, src or self.harness)

    def test_no_optimize_paths_is_refused(self) -> None:
        msg = self._check([])
        self.assertTrue(msg)
        self.assertIn("harness", msg.lower())  # says what the shape should be

    def test_a_path_outside_the_workspace_is_refused(self) -> None:
        fd, outside = tempfile.mkstemp(suffix=".py")
        os.close(fd)
        self.addCleanup(os.remove, outside)
        self.assertIn("outside the workspace", self._check([outside]) or "")

    def test_the_harness_cannot_be_its_own_subject(self) -> None:
        # THE refusal: it is what makes a self-contained copy inexpressible.
        msg = self._check([self.harness, self.a])
        self.assertTrue(msg)
        self.assertIn("cannot be one of optimize_paths", msg)

    def test_a_missing_path_is_refused(self) -> None:
        self.assertIn("not found", self._check([os.path.join(self.wt, "ghost.py")]) or "")

    def test_the_harness_plus_real_code_shape_is_accepted(self) -> None:
        # The oracle: the shape neither recorded session produced.
        self.assertIsNone(self._check([self.a, self.b]))


class TreeAtomicityTests(_Workspace):
    """A restore puts back a state that was measured — never a mix of runs."""

    def _snap(self, message: str) -> str:
        from _lib import tree_snapshot
        sid = tree_snapshot.snapshot(
            os.path.join(self.wt, "proxy_bench", "opt.git"), self.wt,
            [self.a, self.b], message)
        self.assertTrue(sid, "snapshot failed")
        return sid

    def _restore(self, sid: str) -> bool:
        from _lib import tree_snapshot
        return tree_snapshot.restore(
            os.path.join(self.wt, "proxy_bench", "opt.git"), self.wt, [self.a, self.b], sid)

    def _scenario(self) -> None:
        """baseline -> an accepted run -> a regression touching ONE file."""
        self.baseline = self._snap("baseline")
        self._write(self.a, "A1\n")
        self._write(self.b, "B1\n")
        self.accepted = self._snap("run accepted")
        self._write(self.a, "A2_REGRESSION\n")   # only one of the two changes

    def test_reset_to_best_restores_every_tracked_file(self) -> None:
        self._scenario()
        self.assertTrue(self._restore(self.accepted))
        # Both files at the accepted run. Restoring only the file that changed would
        # leave B at B1 by luck here — and assemble a never-measured pair as soon as the
        # regression had touched B instead.
        self.assertEqual((self._read(self.a), self._read(self.b)), ("A1", "B1"))

    def test_reset_goes_all_the_way_back(self) -> None:
        self._scenario()
        self.assertTrue(self._restore(self.baseline))
        self.assertEqual((self._read(self.a), self._read(self.b)), ("A0", "B0"))

    def test_a_snapshot_is_immune_to_edits_made_after_it(self) -> None:
        """`source_at_launch`'s property, generalised: what ran is what is kept."""
        launch = self._snap("launch")
        self._write(self.a, "EDITED_MID_FLIGHT\n")
        self.assertTrue(self._restore(launch))
        self.assertEqual(self._read(self.a), "A0")

    def test_the_fallback_has_the_same_semantics(self) -> None:
        # Same suite with git unavailable: a machine without git must degrade, not lose
        # the invariant.
        from _lib import tree_snapshot
        real = tree_snapshot.git_available
        tree_snapshot.git_available = lambda: False
        self.addCleanup(setattr, tree_snapshot, "git_available", real)
        self._scenario()
        self.assertTrue(self._restore(self.accepted))
        self.assertEqual((self._read(self.a), self._read(self.b)), ("A1", "B1"))

    def test_the_fingerprint_notices_a_changed_tree(self) -> None:
        from _lib import tree_snapshot
        before = tree_snapshot.fingerprint(self.wt, [self.a, self.b])
        self._write(self.b, "CHANGED\n")
        self.assertNotEqual(before, tree_snapshot.fingerprint(self.wt, [self.a, self.b]))


class StoreLocationTests(unittest.TestCase):
    def test_the_default_is_under_the_workspace(self) -> None:
        # It used to be ~/.cache/proxy_bench: deleting a project left the registry and
        # an "in progress" optimisation behind, and a fresh start silently resumed it.
        import importlib
        wt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, wt, True)
        os.environ["MCP_FILES_ROOT"] = wt
        os.environ.pop("MIMIR_PROXY_BENCH_DIR", None)
        self.addCleanup(os.environ.pop, "MCP_FILES_ROOT", None)
        from _lib import store
        importlib.reload(store)
        self.assertEqual(store.cache_dir(), os.path.join(wt, "proxy_bench"))

    def test_the_env_override_still_wins(self) -> None:
        import importlib
        os.environ["MIMIR_PROXY_BENCH_DIR"] = "/tmp/explicit_store"
        self.addCleanup(os.environ.pop, "MIMIR_PROXY_BENCH_DIR", None)
        from _lib import store
        importlib.reload(store)
        self.assertEqual(store.cache_dir(), "/tmp/explicit_store")


class GuidanceTests(unittest.TestCase):
    """The two sentences that produced the duplicate must stay gone.

    Not style: `init`'s reply said "Modify '<proxy_source_path>' between runs" and the
    skill's last rule said "Modify only the proxy source file". Together they pointed the
    model at the harness as the thing to edit, and making that the thing being optimized
    is exactly the copy. A test naming them beats a re-read.
    """

    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    def _tree_text(self) -> str:
        chunks = []
        for sub in ("servers/proxy", "skills/proxy-optimize"):
            for dirpath, _dirs, files in os.walk(os.path.join(self.ROOT, sub)):
                for f in files:
                    if f.endswith((".py", ".md")):
                        with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                            chunks.append(fh.read())
        return "\n".join(chunks)

    def test_the_harness_is_never_named_as_the_thing_to_edit(self) -> None:
        text = self._tree_text()
        for phrase in ("Modify only the proxy source file",
                       "between runs to test new implementations"):
            self.assertNotIn(phrase, text, f"the sentence that produced the copy is back: {phrase!r}")

    def test_the_skill_teaches_optimize_paths(self) -> None:
        with open(os.path.join(self.ROOT, "skills/proxy-optimize/SKILL.md"), encoding="utf-8") as fh:
            skill = fh.read()
        self.assertIn("optimize_paths", skill)
        self.assertIn("clean", skill)


class ProxyCleanCommandTests(unittest.TestCase):
    """`/proxy clean <name>` — housekeeping the user can do without asking the model.

    `clean` is reachable as a tool op, but a person who wants to start an optimisation
    over should not have to ask the model to tidy up first. Before this the only recourse
    was `rm -rf` on a store whose path nobody has a reason to know — which is how a
    deleted project came back with a finished checklist and an optimisation still marked
    "in progress".
    """

    def _run(self, query: str, payload: dict | None):
        import asyncio
        from unittest import mock
        from mimir.client.ui.cli import chat_commands

        async def _fake(agent, tool, arguments):
            self.assertEqual(tool, "proxy_manage")
            self.assertEqual(arguments.get("op"), "clean")
            self.assertTrue(arguments.get("confirm"))
            return payload

        with mock.patch.object(chat_commands, "_call_platform_tool", _fake):
            return asyncio.run(chat_commands.handle_chat_command(
                query=query, mode="agent", thinking=False, streaming=False,
                batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
                set_streaming=lambda v: None, set_batch_mode=lambda v: None,
                agent=object(),
            ))

    def test_it_reports_what_was_removed_and_what_survived(self) -> None:
        handled, msg = self._run("/proxy clean wave2d", {
            "status": "ok",
            "removed": ["runs", "optimisation state"],
            "kept": ["sealed references wave2d_ref — shared with suites"],
        })
        self.assertTrue(handled)
        self.assertIn("wave2d", msg)
        self.assertIn("runs", msg)
        # The surviving half is the answer to "I deleted everything and it still
        # remembers" — printing only what was removed would reproduce the confusion.
        self.assertIn("kept", msg)
        self.assertIn("sealed references", msg)

    def test_a_disconnected_proxy_server_says_so(self) -> None:
        handled, msg = self._run("/proxy clean wave2d", None)
        self.assertTrue(handled)
        self.assertIn("not connected", msg)

    def test_usage_without_a_name(self) -> None:
        from mimir.client.ui.cli import chat_commands
        import asyncio
        handled, msg = asyncio.run(chat_commands.handle_chat_command(
            query="/proxy", mode="agent", thinking=False, streaming=False,
            batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
            set_streaming=lambda v: None, set_batch_mode=lambda v: None, agent=object()))
        self.assertTrue(handled)
        self.assertIn("/proxy clean <name>", msg)

    def test_the_command_is_listed_in_help(self) -> None:
        from mimir.client.ui.cli import chat_commands
        import asyncio
        _handled, msg = asyncio.run(chat_commands.handle_chat_command(
            query="/help", mode="agent", thinking=False, streaming=False,
            batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
            set_streaming=lambda v: None, set_batch_mode=lambda v: None))
        self.assertIn("/proxy clean", msg)


if __name__ == "__main__":
    unittest.main()
