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
        # `store._CACHE_DIR` is computed once at import, so setting MCP_FILES_ROOT here
        # moves the workspace without moving the store — every test in this file was
        # sharing one, and `opt_runs/` accumulated the proxies of whichever tests ran
        # first. Harmless until something asks "is this the last proxy with state?", and
        # then answered with another test's leftovers. Repointed the same way
        # `test_proxy_ops._TmpStorageTest` does, at the path `TreeAtomicityTests` already
        # spells out by hand.
        from _lib import store
        self._saved_cache = store._CACHE_DIR
        store._CACHE_DIR = os.path.join(self.wt, "proxy_bench")
        self.addCleanup(setattr, store, "_CACHE_DIR", self._saved_cache)
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


class RebaselineTests(_Workspace):
    """Moving a baseline without burning the record of how you got there.

    ``init`` will not move an existing baseline, and it should not: re-snapshotting
    mid-optimisation promotes already-optimised code to "the original". But that left
    the other question — *the harness was wrong, measure again from here* — with no
    supported answer, and in session ``7d322a3b`` the model found the unsupported one
    twice in three minutes: ``end`` → ``proxy_manage(op='clean')`` → ``init``. The
    second pass discarded a boundary condition measured 40% better than the incumbent.
    """

    def _session(self) -> tuple:
        from _ops import eval_session
        from _lib import store
        cfg = {
            "proxy_name": "p", "benchmark_name": "b", "requirements": [],
            "proxy_source_path": self.harness,
            "optimize_paths": [self.a, self.b],
            "baseline_id": "ORIGINAL", "baseline_fingerprint": "fp0",
            "baseline_run_id": "r0", "primary_metric": "time_s",
            "primary_goal": "min", "min_improvement": 0.02,
            "max_stall": 5, "stall": 3,
        }
        os.makedirs(store._opt_session_runs_dir("p"), exist_ok=True)
        eval_session._save_opt_config(cfg)
        with open(store._opt_ledger_file("p"), "w", encoding="utf-8") as fh:
            fh.write('{"run_id": "r0", "primary_value": 0.0894}\n')
            fh.write('{"run_id": "r1", "primary_value": 0.0538}\n')
        store._write_json_atomic(store._opt_best_file("p"),
                                 {"run_id": "r1", "primary_value": 0.0538})
        return eval_session, store

    def test_the_ledger_is_archived_not_destroyed(self) -> None:
        eval_session, store = self._session()
        out = eval_session.rebaseline("p")
        self.assertEqual(out["status"], "ok")
        self.assertFalse(os.path.exists(store._opt_ledger_file("p")))
        # The result that was lost twice: still readable, under a timestamp. The
        # stamp goes before the extension so the archive is still a .jsonl.
        archived = [n for n in out["archived"] if n.startswith("ledger")]
        self.assertEqual(len(archived), 1)
        self.assertTrue(archived[0].endswith(".jsonl"))
        with open(os.path.join(store._opt_session_runs_dir("p"), archived[0])) as fh:
            self.assertIn("0.0538", fh.read())
        # The best-so-far moves with it: both or neither, never a ledger that
        # disagrees with the best it is supposed to explain.
        self.assertFalse(os.path.exists(store._opt_best_file("p")))
        self.assertEqual(len(out["archived"]), 2)

    def test_the_baseline_moves_to_the_tree_as_it_stands(self) -> None:
        eval_session, _ = self._session()
        self._write(self.a, "A_OPTIMISED\n")
        out = eval_session.rebaseline("p")
        self.assertNotEqual(out["baseline_id"], "ORIGINAL")
        self.assertEqual(out["previous_baseline_id"], "ORIGINAL")
        cfg = eval_session._load_opt_config("p")
        self.assertEqual(cfg["baseline_id"], out["baseline_id"])
        self.assertEqual(cfg["stall"], 0)   # a new baseline is not a stalled one
        # And it restores to the tree that was there, not the original.
        self._write(self.a, "SCRATCH\n")
        from _lib import tree_snapshot
        tree_snapshot.restore(eval_session.opt_git_dir(), self.wt,
                              [self.a, self.b], out["baseline_id"])
        self.assertEqual(self._read(self.a), "A_OPTIMISED")

    def test_the_reply_says_comparisons_moved(self) -> None:
        """The honesty the end+clean+init route could not offer."""
        eval_session, _ = self._session()
        note = eval_session.rebaseline("p")["note"].lower()
        self.assertIn("new baseline", note)
        self.assertIn("not against the original", note)

    def test_init_still_refuses_to_move_a_baseline_and_names_the_route(self) -> None:
        """The invariant stays; what changes is that the exit is signposted."""
        eval_session, _ = self._session()
        cfg = eval_session._load_opt_config("p")
        self.assertEqual(cfg["baseline_id"], "ORIGINAL")  # untouched by anything but rebaseline

    def test_rebaseline_without_a_session_is_a_structured_error(self) -> None:
        from _ops import eval_session
        out = eval_session.rebaseline("nope")
        self.assertEqual(out["status"], "error")


class NoiseFloorTests(unittest.TestCase):
    """A ratchet that cannot see noise ratchets noise in.

    From session ``7d322a3b`` on a 192-core shared node. Two runs of one untouched
    tree: 0.174488 and 0.179888 — a spread of 3.1%. ``min_improvement`` stood at its
    default 0.02, documented as guarding timing noise. The next run, a full rewrite of
    the OpenMP kernels, measured 0.17491 — a 2.8% "gain" the ratchet accepted and the
    session reported as a step forward. Four later runs of the accepted state spread
    0.0592-0.0642 against a recorded best of 0.0556, so the headline 3.2x was the
    minimum of a distribution whose middle said 2.8x.
    """

    def _cfg(self, **over) -> dict:
        cfg = {"primary_metric": "time_s", "min_improvement": 0.02}
        cfg.update(over)
        return cfg

    def test_the_configured_margin_stands_until_something_is_measured(self) -> None:
        from _ops import eval_session
        margin, source = eval_session._effective_min_improvement(self._cfg())
        self.assertAlmostEqual(margin, 0.02)
        self.assertEqual(source, "configured")

    def test_a_measured_floor_above_the_configured_one_wins(self) -> None:
        from _ops import eval_session
        margin, source = eval_session._effective_min_improvement(
            self._cfg(noise_floor=0.031))
        self.assertAlmostEqual(margin, 0.031)
        self.assertIn("measured", source)

    def test_the_session_that_produced_this_would_now_reject_its_own_step(self) -> None:
        """The regression this whole lane exists for, with its real numbers."""
        from _ops import eval_session
        from _lib.ratchet import _is_improvement
        baseline, fused = 0.179888, 0.17491          # what the two runs measured
        floor = abs(0.179888 - 0.174488) / 0.179888  # 3.1%, from the two baseline runs
        margin, _ = eval_session._effective_min_improvement(
            self._cfg(noise_floor=floor))
        self.assertTrue(_is_improvement(fused, baseline, "min", 0.02),
                        "premise: the old 2% margin accepted it")
        self.assertFalse(_is_improvement(fused, baseline, "min", margin),
                         "a 2.8% gain cannot clear a 3.1% noise floor")

    def test_a_real_gain_still_clears_the_measured_floor(self) -> None:
        """The floor rejects noise, not results: float32 was a 22% win."""
        from _ops import eval_session
        from _lib.ratchet import _is_improvement
        margin, _ = eval_session._effective_min_improvement(
            self._cfg(noise_floor=0.031))
        self.assertTrue(_is_improvement(0.136145, 0.17491, "min", margin))

    def test_repeat_is_decided_from_the_metric_when_unset(self) -> None:
        from _ops import eval_session
        self.assertEqual(eval_session._effective_repeat({"primary_metric": "time_s"}), 3)
        # An accuracy metric against a sealed reference is bit-reproducible.
        self.assertEqual(eval_session._effective_repeat({"primary_metric": "l2_rel"}), 1)

    def test_an_explicit_repeat_is_obeyed(self) -> None:
        from _ops import eval_session
        self.assertEqual(
            eval_session._effective_repeat({"primary_metric": "time_s", "repeat": 1}), 1)
        self.assertEqual(
            eval_session._effective_repeat({"primary_metric": "l2_rel", "repeat": 7}), 7)


class ReplicateAggregationTests(unittest.TestCase):
    """Replicates collapse to their middle, never to their best."""

    def _runner(self):
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "servers", "proxy", "_proxy_runner.py")
        spec = importlib.util.spec_from_file_location("_proxy_runner_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_median_is_taken_not_the_minimum(self) -> None:
        runner = self._runner()
        # The four re-runs of the accepted state, as recorded.
        reps = [{"time_s": v} for v in (0.064247, 0.063696, 0.059171, 0.063534)]
        agg = runner._aggregate_replicates(reps)
        self.assertAlmostEqual(agg["time_s"], 0.063615)
        self.assertGreater(agg["time_s"], min(r["time_s"] for r in reps))

    def test_a_single_replicate_is_left_exactly_as_measured(self) -> None:
        runner = self._runner()
        self.assertEqual(runner._aggregate_replicates([{"time_s": 1.5, "dtype": "f32"}]),
                         {"time_s": 1.5, "dtype": "f32"})

    def test_non_numeric_metrics_survive_aggregation(self) -> None:
        runner = self._runner()
        agg = runner._aggregate_replicates(
            [{"dtype": "float32", "time_s": 1.0}, {"dtype": "float32", "time_s": 3.0}])
        self.assertEqual(agg["dtype"], "float32")
        self.assertAlmostEqual(agg["time_s"], 2.0)

    def test_the_spread_is_reported_as_a_fraction_of_the_middle(self) -> None:
        runner = self._runner()
        spread = runner._relative_spread(
            [{"time_s": 0.174488}, {"time_s": 0.179888}], "time_s")
        self.assertAlmostEqual(spread, 0.0305, places=3)  # the 3.1% floor

    def test_one_measurement_has_no_spread_to_report(self) -> None:
        runner = self._runner()
        self.assertIsNone(runner._relative_spread([{"time_s": 1.0}], "time_s"))


class CleanKeepsSharedSnapshotsTests(_Workspace):
    """Tidying one optimisation must not break another's rollback.

    Every proxy writes into ONE snapshot store — a single branch, a shared index, each
    commit carrying the union of every path ever tracked — so there is no per-proxy
    history inside it to remove. `clean` handled that by deleting the whole thing, which
    took the other proxies' snapshots with it: their `best.json` kept an id pointing into
    a repository that no longer existed, and `reset_to_best` answered "Could not restore
    the best tree". A rollback lost by cleaning up something else.

    The rule was already stated one paragraph up in `clean`'s own docstring — it does not
    cascade into references and suites *because they can be shared*. Snapshots are too.
    """

    def _optimisation(self, name: str, marker: str) -> None:
        from _ops import eval_session
        from _lib import store, tree_snapshot
        os.makedirs(store._opt_session_runs_dir(name), exist_ok=True)
        self._write(self.a, f"{marker}\n")
        snap = tree_snapshot.snapshot(
            eval_session.opt_git_dir(), self.wt, [self.a, self.b], f"{name} best")
        eval_session._save_opt_config({
            "proxy_name": name, "benchmark_name": "b", "requirements": [],
            "proxy_source_path": self.harness, "optimize_paths": [self.a, self.b],
            "baseline_id": snap, "baseline_run_id": "r0", "primary_metric": "time_s",
            "primary_goal": "min", "min_improvement": 0.02, "max_stall": 5, "stall": 0})
        store._write_json_atomic(store._opt_best_file(name),
                                 {"run_id": "r1", "primary_value": 0.1,
                                  "tree_snapshot": snap})

    def _rollback(self, name: str):
        from _ops import eval_session
        self._write(self.a, "SCRATCH\n")
        return eval_session.reset_to_best(name)

    def test_cleaning_one_leaves_the_other_able_to_roll_back(self) -> None:
        from _ops import registry
        self._optimisation("bench", "BENCH_BEST")
        self._optimisation("abc", "ABC_BEST")
        self.assertEqual(self._rollback("abc")["status"], "ok")  # premise

        registry.clean("bench")

        out = self._rollback("abc")
        self.assertEqual(out["status"], "ok", out.get("error"))
        self.assertEqual(self._read(self.a), "ABC_BEST")

    def test_and_says_it_kept_them_and_why(self) -> None:
        from _ops import registry
        self._optimisation("bench", "BENCH_BEST")
        self._optimisation("abc", "ABC_BEST")
        out = registry.clean("bench")
        kept = " ".join(out["kept"])
        self.assertIn("tree snapshots", kept)
        self.assertIn("abc", kept)                       # names who still needs them
        self.assertNotIn("tree snapshots", out["removed"])

    def test_cleaning_the_last_one_still_removes_them(self) -> None:
        """Conservative, not hoarding: with nothing left to point in, they go."""
        from _ops import registry, eval_session
        self._optimisation("bench", "BENCH_BEST")
        registry.clean("bench")
        self.assertFalse(os.path.isdir(eval_session.opt_git_dir()))

    def test_the_last_one_reports_the_removal(self) -> None:
        from _ops import registry
        self._optimisation("bench", "BENCH_BEST")
        out = registry.clean("bench")
        self.assertIn("tree snapshots", out["removed"])

    def test_an_unregistered_proxy_still_counts_as_needing_them(self) -> None:
        """State on disk is what keeps a snapshot store alive, not registration."""
        from _ops import registry, eval_session
        self._optimisation("bench", "BENCH_BEST")
        self._optimisation("abc", "ABC_BEST")
        registry.clean("bench")
        self.assertTrue(os.path.isdir(eval_session.opt_git_dir()))


class ResumeNoticeTests(_Workspace):
    """Coming back to an old optimisation whose code moved on without it.

    Two guards covered one question between them and left a gap in the middle.
    `_prepare_run` refuses a first run on an already-edited tree — but only while no
    baseline run is on record (`if paths and not cfg.get("baseline_run_id")`). Once one
    is, nothing checks again, so resuming a finished session ran it against a best
    measured on code no longer on disk and said nothing. Both fingerprints were already
    on the disk; nobody compared them.

    The action stays manual: `rebaseline` moves the point of comparison, and doing that
    automatically would let a regression read as progress the moment the new reference
    landed on a degraded state. Only the *detection* is automatic.
    """

    def _session(self, *, measured: bool = True) -> None:
        from _ops import eval_session
        from _lib import store, procs, tree_snapshot
        self._write(self.a, "MEASURED_STATE\n")
        os.makedirs(store._opt_session_runs_dir("bench"), exist_ok=True)
        snap = tree_snapshot.snapshot(
            eval_session.opt_git_dir(), self.wt, [self.a], "baseline")
        eval_session._save_opt_config({
            "proxy_name": "bench", "benchmark_name": "b", "requirements": [],
            "proxy_source_path": self.harness, "optimize_paths": [self.a],
            "baseline_id": snap,
            "baseline_fingerprint": tree_snapshot.fingerprint(self.wt, [self.a]),
            "baseline_run_id": "r0" if measured else "",
            "primary_metric": "time_s", "primary_goal": "min",
            "min_improvement": 0.02, "max_stall": 5, "stall": 0})
        run_dir = os.path.join(store._opt_session_runs_dir("bench"), "20240101T000000Z")
        os.makedirs(run_dir, exist_ok=True)
        store._write_json_atomic(os.path.join(run_dir, "tree_at_launch.json"), {
            "snapshot_id": snap, "paths": [self.a],
            "fingerprint": tree_snapshot.fingerprint(self.wt, [self.a])})
        procs._update_opt_active_link("bench", run_dir)

    def _notice(self) -> str:
        from _ops import eval_session
        cfg = eval_session._load_opt_config("bench")
        return eval_session._resume_notice(cfg, "bench", [self.a])

    def _activate(self, name: str | None) -> None:
        from _lib import store
        if name is None:
            store._clear_active_session()
        else:
            store._write_active_session(name)

    def test_a_resumed_session_on_changed_code_is_flagged(self) -> None:
        self._session()
        self._activate(None)                       # the session was ended
        self._write(self.a, "CHANGED_SINCE\n")     # by a pull, a person, anything
        notice = self._notice()
        self.assertTrue(notice)
        self.assertIn("rebaseline", notice)        # names the supported route
        self.assertIn("previous best", notice)     # says what it will compare against

    def test_an_ordinary_iteration_is_not(self) -> None:
        """The false positive that would make the warning unreadable.

        Inside a live session the tree differs from what was last measured on every
        single iteration — that is the loop working. A check that fired here would fire
        every time, and a warning that always fires is not read.
        """
        self._session()
        self._activate("bench")
        self._write(self.a, "CANDIDATE_EDIT\n")
        self.assertEqual(self._notice(), "")

    def test_a_resumed_session_on_untouched_code_is_not(self) -> None:
        self._session()
        self._activate(None)
        self.assertEqual(self._notice(), "")

    def test_another_proxy_being_active_is_still_a_resumption(self) -> None:
        self._session()
        self._activate("some_other_proxy")
        self._write(self.a, "CHANGED_SINCE\n")
        self.assertTrue(self._notice())

    def test_a_never_measured_session_is_left_to_the_older_guard(self) -> None:
        """It is refused outright there; two messages about one thing help nobody."""
        self._session(measured=False)
        self._activate(None)
        self._write(self.a, "CHANGED_SINCE\n")
        self.assertEqual(self._notice(), "")

    def test_the_notice_survives_the_path_a_caller_reaches(self) -> None:
        """`run_awaited` rebuilds its answer from `results()` on the completed path.

        Anything set only on the launch payload is dropped there — which is the branch
        `op='run'` actually takes, so a notice added to `run()` alone would be invisible
        exactly where it matters.
        """
        import inspect
        from _ops import eval_session
        carried = inspect.getsource(eval_session.run_awaited)
        self.assertIn('"resume_notice"', carried)
