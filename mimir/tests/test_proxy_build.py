"""The build step the ratchet runs before it measures (servers/proxy/_lib/build.py).

A Python proxy needs none: the edited file is the file that runs. A compiled one
breaks that identity, and the ratchet fingerprints sources only — so an unbuilt
binary would be measured, and could be *accepted*, against code that never ran.
These tests pin the three properties that close it: the build happens once per
run and before any case, a failed build produces no verdict and no ledger entry,
and a restore leaves sources the build system will notice.

Run:
    python -m unittest mimir.tests.test_proxy_build -v
"""

import json
import os
import sys
import time
import unittest

from mimir.tests._proxy_fixtures import (  # noqa: F401
    _TmpStorageTest, _script, build, eval_session, procs, ratchet, registry,
    server_proxy, store, tree_snapshot,
)

class SpecTests(_TmpStorageTest):
    def test_no_build_cmd_is_no_build(self) -> None:
        self.assertIsNone(build.spec({}))
        self.assertIsNone(build.spec({"build_cmd": "   "}))
        rec = build.run_build({}, self.root, proxy="tiny")
        self.assertEqual(rec["status"], "skipped")
        self.assertFalse(os.path.exists(procs._build_log_path(self.root)))
        self.assertFalse(os.path.exists(build.report_path(self.root)))

    def test_defaults_are_filled_and_the_timeout_is_bounded(self) -> None:
        sp = build.spec({"build_cmd": "make"})
        self.assertEqual(sp["cmd"], "make")
        self.assertEqual(sp["timeout_s"], build._DEFAULT_TIMEOUT_S)
        huge = build.spec({"build_cmd": "make", "build_timeout_s": 10 ** 9})
        self.assertLessEqual(huge["timeout_s"], 6 * 3600.0)


class BuildCwdSpellingTests(_TmpStorageTest):
    """The build runs in the directory it was configured with, symlinks and all.

    Resolving them is a full rebuild on every run: scratch space is normally presented
    through a link (/home/me/proj -> /gpfs/.../proj), CMake bakes the absolute path it
    was configured with into its cache, rules and depfiles, and a build handed the
    other spelling finds every one of them unequal. It then alternates forever with
    whatever the user runs from their own shell.
    """

    def _linked_tree(self) -> tuple[str, str]:
        real = os.path.join(self.root, "real_build")
        os.makedirs(real, exist_ok=True)
        link = os.path.join(self.root, "linked_build")
        if not os.path.islink(link):
            os.symlink(real, link)
        return real, link

    def test_a_symlinked_build_cwd_is_not_resolved(self) -> None:
        real, link = self._linked_tree()
        cwd, error = build._resolve_cwd(link)
        self.assertIsNone(error)
        self.assertEqual(cwd, link)
        self.assertNotEqual(cwd, real)

    def test_the_build_process_itself_sees_the_declared_spelling(self) -> None:
        """The assertion that matters: what the child sees, not what we recorded.

        Popen chdir()s, and getcwd() comes back resolved whichever spelling it was
        handed — so keeping build_cwd unresolved buys nothing on its own. PWD is the
        only channel that carries it, and it used to be inherited: it named the MCP
        server's directory rather than the one the build was running in.
        """
        real, link = self._linked_tree()
        sh = _script(os.path.join(self.root, "b.sh"), 'echo "at=$(pwd)"\n')
        rec = build.run_build({"build_cmd": sh, "build_cwd": link},
                              self.root, proxy="tiny")
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["cwd"], link)
        log = open(procs._build_log_path(self.root)).read()
        self.assertIn(f"at={link}", log)
        self.assertNotIn(f"at={real}", log)

    def test_a_relative_build_cwd_still_resolves_under_the_workspace(self) -> None:
        os.makedirs(os.path.join(self.root, "sub", "build"), exist_ok=True)
        cwd, error = build._resolve_cwd("sub/build")
        self.assertIsNone(error)
        self.assertEqual(cwd, os.path.join(self.root, "sub", "build"))

    def test_a_missing_directory_is_still_refused(self) -> None:
        _cwd, error = build._resolve_cwd(os.path.join(self.root, "nope"))
        self.assertIn("not a directory", error)


class BuildEnvTests(_TmpStorageTest):
    """The environment a build gets, pinned where it matters and recorded always."""

    def test_pinned_variables_reach_the_build(self) -> None:
        sh = _script(os.path.join(self.root, "b.sh"), 'echo "CC=$CC"\n')
        rec = build.run_build(
            {"build_cmd": sh, "build_env": {"CC": "my-special-gcc"}},
            self.root, proxy="tiny")
        self.assertEqual(rec["status"], "ok")
        self.assertIn("CC=my-special-gcc", open(procs._build_log_path(self.root)).read())
        self.assertEqual(rec["env_declared"], {"CC": "my-special-gcc"})

    def test_nothing_pinned_means_the_process_is_launched_as_before(self) -> None:
        sh = _script(os.path.join(self.root, "b.sh"), "true\n")
        rec = build.run_build({"build_cmd": sh}, self.root, proxy="tiny")
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["env_declared"], {})

    def test_the_log_says_whether_ccache_was_reachable(self) -> None:
        """A ccache miss looked exactly like "it rebuilt everything" and left no trace."""
        sh = _script(os.path.join(self.root, "b.sh"), "true\n")
        build.run_build({"build_cmd": sh, "build_env": {"PATH": "/nonexistent"}},
                        self.root, proxy="tiny")
        log = open(procs._build_log_path(self.root)).read()
        self.assertIn("ccache: not on PATH", log)

    def test_a_malformed_variable_name_is_refused_at_registration(self) -> None:
        res = server_proxy.proxy_manage(
            op="register", name="tiny", executable_path=self._make_exe(),
            run_cmd_template="python3 {executable}",
            metadata={"build_cmd": "make", "build_env": {"not a name": "x"}},
            confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("build_env", res["error"])

    def test_a_malformed_environment_is_refused_at_build_time_too(self) -> None:
        rec = build.run_build({"build_cmd": "make", "build_env": {"2BAD": "x"}},
                              self.root, proxy="tiny")
        self.assertEqual(rec["status"], "invalid")
        self.assertIn("build_env", rec["error"])


class ShellShapeTests(_TmpStorageTest):
    """build_cmd is argv, and the refusal must arrive before the build does."""

    def test_a_shell_operator_is_refused_at_registration(self) -> None:
        res = server_proxy.proxy_manage(
            op="register", name="tiny", executable_path=self._make_exe(),
            run_cmd_template="python3 {executable}",
            metadata={"build_cmd": "make -C x && ./y"}, confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("&&", res["error"])
        self.assertIn("wrapper script", res["error"])

    def test_a_shell_operator_is_refused_at_build_time_too(self) -> None:
        # A registry written before the check, or edited by hand, still must not
        # hand execvp() a literal "&&" and fail half an hour later.
        rec = build.run_build({"build_cmd": "make && ./y"}, self.root, proxy="tiny")
        self.assertEqual(rec["status"], "invalid")
        self.assertIn("wrapper script", rec["error"])

    def test_a_plain_command_is_accepted(self) -> None:
        self.assertIsNone(build.check_cmd("make -C /abs/build solver"))


class RunBuildTests(_TmpStorageTest):
    def test_output_is_captured_and_the_code_is_reported(self) -> None:
        sh = _script(os.path.join(self.root, "b.sh"),
                     "echo out; echo err 1>&2; exit 3\n")
        rec = build.run_build({"build_cmd": sh}, self.root, proxy="tiny")
        self.assertEqual(rec["status"], "failed")
        self.assertEqual(rec["returncode"], 3)
        self.assertIsInstance(rec["duration_s"], float)
        log = open(procs._build_log_path(self.root)).read()
        self.assertIn("out", log)
        self.assertIn("err", log)

    def test_a_successful_build_reports_ok(self) -> None:
        sh = _script(os.path.join(self.root, "b.sh"), "echo building\n")
        rec = build.run_build({"build_cmd": sh}, self.root, proxy="tiny")
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["returncode"], 0)

    def test_a_missing_build_program_is_a_launch_error_not_a_crash(self) -> None:
        rec = build.run_build(
            {"build_cmd": os.path.join(self.root, "does-not-exist")},
            self.root, proxy="tiny")
        self.assertEqual(rec["status"], "launch_error")

    def test_a_timeout_kills_the_whole_group(self) -> None:
        """make -j forks; killing only the shell would leave compilers running."""
        marker = os.path.join(self.root, "grandchild.pid")
        sh = _script(os.path.join(self.root, "b.sh"),
                     f"sh -c 'echo $$ > {marker}; sleep 30' &\n"
                     "sleep 30\n")
        rec = build.run_build({"build_cmd": sh, "build_timeout_s": 1},
                              self.root, proxy="tiny")
        self.assertEqual(rec["status"], "timeout")
        time.sleep(0.5)
        pid = int(open(marker).read().strip())
        with self.assertRaises(OSError):
            os.kill(pid, 0)

    def test_a_bad_build_cwd_is_named_rather_than_guessed(self) -> None:
        rec = build.run_build(
            {"build_cmd": "make", "build_cwd": os.path.join(self.root, "nope")},
            self.root, proxy="tiny")
        self.assertEqual(rec["status"], "invalid")
        self.assertIn("build_cwd", rec["error"])


class BuildsForTests(_TmpStorageTest):
    """What gets built must be what gets measured."""

    def _suite(self, *proxies: str) -> dict:
        return {"cases": [{"case_id": f"c{i}", "proxy_name": p}
                          for i, p in enumerate(proxies)]}

    def test_one_build_per_distinct_entry_in_first_use_order(self) -> None:
        reg = {"a": {"build_cmd": "make a"}, "b": {"build_cmd": "make b"}}
        got = build.builds_for(reg, self._suite("b", "a", "b"), "a", reg["a"])
        self.assertEqual([n for n, _ in got], ["b", "a"])

    def test_two_proxies_sharing_a_build_command_build_once(self) -> None:
        reg = {"a": {"build_cmd": "make all"}, "b": {"build_cmd": "make all"}}
        got = build.builds_for(reg, self._suite("a", "b"), "a", reg["a"])
        self.assertEqual(len(got), 1)

    def test_a_case_naming_an_unknown_proxy_falls_back_like_the_runner(self) -> None:
        # Mirrors reg.get(case_proxy, entry) in _proxy_runner: resolving it any
        # other way would build a proxy the run does not execute.
        fallback = {"build_cmd": "make fallback"}
        got = build.builds_for({}, self._suite("ghost"), "a", fallback)
        self.assertEqual([e["build_cmd"] for _, e in got], ["make fallback"])

    def test_a_suite_of_proxies_without_builds_needs_none(self) -> None:
        reg = {"a": {}, "b": {}}
        self.assertEqual(build.builds_for(reg, self._suite("a", "b"), "a", reg["a"]), [])


class ReportTests(_TmpStorageTest):
    def test_the_report_carries_the_first_failure(self) -> None:
        payload = build.write_report(self.root, [
            {"proxy": "a", "status": "ok", "duration_s": 1.5},
            {"proxy": "b", "status": "failed", "duration_s": 2.0,
             "error": "build exited 2"},
        ])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["total_duration_s"], 3.5)
        self.assertIn("b", build.failure_summary(payload))
        self.assertEqual(build.read_report(self.root)["status"], "failed")

    def test_an_all_ok_report_is_ok(self) -> None:
        payload = build.write_report(self.root, [{"proxy": "a", "status": "ok",
                                                  "duration_s": 0.5}])
        self.assertEqual(payload["status"], "ok")


class RegistrationTests(_TmpStorageTest):
    def test_an_unbuilt_executable_is_accepted_when_a_build_produces_it(self) -> None:
        exe = os.path.join(self.root, "build", "solver")
        res = server_proxy.proxy_manage(
            op="register", name="cpp", executable_path=exe,
            run_cmd_template="{executable} {param_file}",
            metadata={"build_cmd": "make -C /abs solver"}, confirm=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        self.assertEqual(res["executable_not_yet_built"], exe)

    def test_a_missing_executable_is_still_refused_without_a_build(self) -> None:
        res = server_proxy.proxy_manage(
            op="register", name="cpp", executable_path=os.path.join(self.root, "nope"),
            run_cmd_template="{executable}", confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("not found", res["error"])

    def test_every_metadata_key_is_documented(self) -> None:
        """The drift this whole change exists to undo, caught mechanically.

        A registration field the tool description does not mention is a field
        nobody uses: build_cmd sat in the registry for the life of the server,
        documented as one word in a list, and was read as decoration.
        """
        tool = getattr(server_proxy.proxy_manage, "fn", server_proxy.proxy_manage)
        doc = tool.__doc__ or ""
        for key in registry._METADATA_DEFAULTS:
            self.assertIn(key, doc,
                          f"metadata key {key!r} is accepted but undocumented")

    def test_the_readme_says_the_build_is_run_not_documented(self) -> None:
        self._register("tiny", metadata={"build_cmd": "make tiny"})
        readme = server_proxy.proxy_get(op="proxy", name="tiny")["readme"]
        self.assertIn("## Build", readme)
        self.assertIn("before any case is measured", readme)


class FailedBuildTests(_TmpStorageTest):
    """A build failure must be an error, not a run that 'may still be in progress'."""

    def _session_with_failed_build(self) -> str:
        self._register("tiny")
        store._save_opt_config({
            "proxy_name": "tiny", "benchmark_name": "suite",
            "proxy_source_path": self._make_exe("harness.py"),
            "optimize_paths": [self._tracked()],
            "requirements": [], "primary_metric": "time_s", "primary_goal": "min",
        }, "tiny")
        run_dir = procs._new_run_dir(store._opt_session_runs_dir("tiny"), "failed")
        build.write_report(run_dir, [{"proxy": "tiny", "status": "failed",
                                      "returncode": 2, "duration_s": 4.0,
                                      "error": "build exited 2"}])
        with open(procs._build_log_path(run_dir), "w") as fh:
            fh.write("solver.cpp:12:5: error: expected ';'\n")
        procs._update_opt_active_link("tiny", run_dir)
        return run_dir

    def test_results_reports_the_build_failure_with_its_log(self) -> None:
        self._session_with_failed_build()
        res = server_proxy.proxy_eval_status(op="results", proxy_name="tiny")
        self.assertEqual(res["status"], "error")
        self.assertIn("Build failed", res["error"])
        self.assertIn("expected ';'", res["build_log_tail"])
        self.assertEqual(res["state"], "crashed")
        self.assertNotIn("verdict", res)

    def test_a_failed_build_is_never_reported_as_in_progress(self) -> None:
        self._session_with_failed_build()
        res = server_proxy.proxy_eval_status(op="results", proxy_name="tiny")
        self.assertNotIn("may still be in progress",
                         json.dumps(res, default=str))

    def test_a_failed_build_leaves_no_verdict_and_no_ledger(self) -> None:
        self._session_with_failed_build()
        server_proxy.proxy_eval_status(op="results", proxy_name="tiny")
        self.assertIsNone(ratchet._load_best("tiny"))
        self.assertFalse(os.path.exists(store._opt_ledger_file("tiny")))
        cfg = store._load_opt_config("tiny")
        self.assertEqual(cfg.get("stall", 0), 0)
        self.assertFalse(cfg.get("baseline_run_id"))


class LongBuildAdviceTests(_TmpStorageTest):
    """Advice keyed on a measured number, never on a guess about the project."""

    def test_a_long_build_is_told_to_detach_next_time(self) -> None:
        rec = eval_session._long_build_note(400.0)
        self.assertIn("background=True", rec)
        self.assertIn("400s", rec)

    def test_a_trivial_build_says_nothing(self) -> None:
        self.assertEqual(eval_session._long_build_note(0.4), "")
        self.assertEqual(eval_session._long_build_note(0.0), "")


class RestoreMtimeTests(_TmpStorageTest):
    """copy2 preserved the snapshot's mtime, which is what make reads."""

    def test_a_restored_file_is_newer_than_the_artifact_built_from_it(self) -> None:
        src = os.path.join(self.root, "solver.cpp")
        with open(src, "w") as fh:
            fh.write("int best(){return 1;}\n")
        old = time.time() - 10_000
        os.utime(src, (old, old))

        git_dir = os.path.join(self.root, "opt.git")
        snap = tree_snapshot._copy_snapshot(git_dir, self.root, [src], "best")
        self.assertIsNotNone(snap)

        with open(src, "w") as fh:
            fh.write("int regression(){return 2;}\n")
        artifact_built_at = time.time()
        time.sleep(0.01)

        self.assertIsNotNone(tree_snapshot._copy_restore(git_dir, self.root, [src], snap))
        self.assertEqual(open(src).read(), "int best(){return 1;}\n")
        self.assertGreater(os.path.getmtime(src), artifact_built_at)


class SelectiveRestoreTests(_TmpStorageTest):
    """A restore writes only what differs, because writing means "recompile this".

    A fresh mtime is how a restored file tells `make` it is new. Handing that to a file
    whose bytes never changed is a false statement: the artifact beside it was built
    from exactly those bytes. Both backends used to hand it to the whole tracked set, so
    undoing a one-file edit cost a rebuild of every tracked file — and of everything
    downstream of a header among them. These pin the narrowing, in both backends, and
    the invariant it must not cost: the tree still ends up exactly at the snapshot.
    """

    def _pair(self) -> tuple[str, str]:
        a = os.path.join(self.root, "a.cpp")
        b = os.path.join(self.root, "b.cpp")
        for path, body in ((a, "int a(){return 1;}\n"), (b, "int b(){return 2;}\n")):
            with open(path, "w") as fh:
                fh.write(body)
        return a, b

    def _backends(self):
        """Both restore paths under one name: git, and the copy fallback.

        A machine without git runs the fallback, so an incrementality property that
        holds in only one of them holds where it is not needed.
        """
        git_dir = os.path.join(self.root, "opt.git")

        def via_git(paths, snap_of):
            sid = tree_snapshot._git_snapshot(git_dir, self.root, paths, snap_of)
            return git_dir, sid, tree_snapshot._git_restore

        def via_copy(paths, snap_of):
            sid = tree_snapshot._copy_snapshot(git_dir, self.root, paths, snap_of)
            return git_dir, sid, tree_snapshot._copy_restore

        return (("git", via_git), ("copy", via_copy))

    def test_an_untouched_file_keeps_its_mtime(self) -> None:
        """The property the whole change exists for."""
        for name, make in self._backends():
            with self.subTest(backend=name):
                a, b = self._pair()
                old = time.time() - 10_000
                os.utime(a, (old, old))
                os.utime(b, (old, old))
                _gd, sid, restore = make([a, b], "best")
                self.assertIsNotNone(sid, f"{name} snapshot failed")

                b_mtime = os.path.getmtime(b)
                with open(a, "w") as fh:          # only a changes
                    fh.write("int a(){return 99;}\n")

                rewritten = restore(_gd, self.root, [a, b], sid)
                self.assertEqual(rewritten, ["a.cpp"])
                # a is back and looks new to the build system...
                self.assertEqual(open(a).read(), "int a(){return 1;}\n")
                self.assertGreater(os.path.getmtime(a), old + 1)
                # ...and b was never written, so nothing downstream of it rebuilds.
                self.assertEqual(os.path.getmtime(b), b_mtime)

    def test_a_tree_already_in_that_state_is_not_touched_and_is_not_a_failure(self) -> None:
        for name, make in self._backends():
            with self.subTest(backend=name):
                a, b = self._pair()
                _gd, sid, restore = make([a, b], "best")
                mtimes = (os.path.getmtime(a), os.path.getmtime(b))

                rewritten = restore(_gd, self.root, [a, b], sid)
                # An empty list is a success. `if not restore(...)` is the one way to
                # read this wrong, which is why callers test `is None`.
                self.assertEqual(rewritten, [])
                self.assertIsNotNone(rewritten)
                self.assertEqual((os.path.getmtime(a), os.path.getmtime(b)), mtimes)

    def test_a_file_deleted_from_disk_is_restored(self) -> None:
        """Where "identical" has no meaning, the file must come back anyway."""
        for name, make in self._backends():
            with self.subTest(backend=name):
                a, b = self._pair()
                _gd, sid, restore = make([a, b], "best")
                os.remove(a)

                rewritten = restore(_gd, self.root, [a, b], sid)
                self.assertEqual(rewritten, ["a.cpp"])
                self.assertEqual(open(a).read(), "int a(){return 1;}\n")

    def test_an_edit_in_the_same_tick_as_a_restore_is_still_seen(self) -> None:
        """Editing right after reset_to_best is the loop, not a corner case.

        A stat-based comparison answers "identical" here: the edit lands in the same
        filesystem tick as the restore that preceded it, so size and mtime are
        unchanged while the content is not. Getting this wrong leaves a file on disk
        that is not what the snapshot says — the one direction this must never fail in.
        """
        for name, make in self._backends():
            with self.subTest(backend=name):
                a, _b = self._pair()
                _gd, sid, restore = make([a], "best")
                restore(_gd, self.root, [a], sid)
                with open(a, "w") as fh:        # same size, same tick
                    fh.write("int a(){return 9;}\n")
                self.assertEqual(restore(_gd, self.root, [a], sid), ["a.cpp"])
                self.assertEqual(open(a).read(), "int a(){return 1;}\n")

    def test_the_tree_still_ends_up_exactly_at_the_snapshot(self) -> None:
        """Narrowing the writes must not narrow the guarantee."""
        for name, make in self._backends():
            with self.subTest(backend=name):
                a, b = self._pair()
                _gd, sid, restore = make([a, b], "best")
                before = tree_snapshot.fingerprint(self.root, [a, b])
                for path in (a, b):
                    with open(path, "w") as fh:
                        fh.write("regression\n")

                rewritten = restore(_gd, self.root, [a, b], sid)
                self.assertEqual(sorted(rewritten), ["a.cpp", "b.cpp"])
                self.assertEqual(tree_snapshot.fingerprint(self.root, [a, b]), before)


class RunnerPhaseTests(_TmpStorageTest):
    """The phase the runner calls, tested without launching a whole run."""

    def _runner(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_proxy_runner_under_test",
            os.path.join(os.path.dirname(os.path.dirname(store.__file__)),
                         "_proxy_runner.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_a_failed_build_stops_the_phase_and_writes_the_report(self) -> None:
        runner = self._runner()
        sh = _script(os.path.join(self.root, "b.sh"), "exit 1\n")
        reg = {"tiny": {"build_cmd": sh, "executable_path": self._make_exe()}}
        suite = {"cases": [{"case_id": "c0", "proxy_name": "tiny"}]}
        ok = runner._build_phase(build, tree_snapshot, reg, suite, "tiny",
                                 reg["tiny"], self.root, {})
        self.assertFalse(ok)
        self.assertEqual(build.read_report(self.root)["status"], "failed")

    def test_a_build_that_leaves_no_executable_fails_the_run(self) -> None:
        runner = self._runner()
        sh = _script(os.path.join(self.root, "b.sh"), "true\n")
        reg = {"tiny": {"build_cmd": sh,
                        "executable_path": os.path.join(self.root, "never-made")}}
        suite = {"cases": [{"case_id": "c0", "proxy_name": "tiny"}]}
        ok = runner._build_phase(build, tree_snapshot, reg, suite, "tiny",
                                 reg["tiny"], self.root, {})
        self.assertFalse(ok)
        rec = build.read_report(self.root)["builds"][-1]
        self.assertIn("still missing", rec["error"])

    def test_a_build_that_rewrites_a_tracked_source_fails_the_run(self) -> None:
        runner = self._runner()
        tracked = self._tracked()
        sh = _script(os.path.join(self.root, "b.sh"),
                     f"echo 'GENERATED = 2' > {tracked}\n")
        exe = self._make_exe()
        reg = {"tiny": {"build_cmd": sh, "executable_path": exe}}
        suite = {"cases": [{"case_id": "c0", "proxy_name": "tiny"}]}
        run_dir = procs._new_run_dir(store._opt_session_runs_dir("tiny"), "g")
        with open(os.path.join(run_dir, "tree_at_launch.json"), "w") as fh:
            json.dump({"paths": [tracked],
                       "fingerprint": tree_snapshot.fingerprint(self.root, [tracked])},
                      fh)
        ok = runner._build_phase(build, tree_snapshot, reg, suite, "tiny",
                                 reg["tiny"], run_dir, {"optimize_paths": [tracked]})
        self.assertFalse(ok)
        self.assertIn("optimize_paths",
                      build.read_report(run_dir)["builds"][-1]["error"])

    def test_a_clean_build_lets_the_run_proceed(self) -> None:
        runner = self._runner()
        exe = self._make_exe()
        sh = _script(os.path.join(self.root, "b.sh"), "echo ok\n")
        reg = {"tiny": {"build_cmd": sh, "executable_path": exe}}
        suite = {"cases": [{"case_id": "c0", "proxy_name": "tiny"}]}
        ok = runner._build_phase(build, tree_snapshot, reg, suite, "tiny",
                                 reg["tiny"], self.root, {})
        self.assertTrue(ok)
        self.assertEqual(build.read_report(self.root)["status"], "ok")


class SplitPhaseTests(_TmpStorageTest):
    """The build and the measurement as two jobs on two machines.

    A node dedicated to GPU simulation is not a node to compile on, so an eval run
    can be split in two: one job builds, a second measures, both over the same run
    directory. What these pin is the seam — the run half must not rebuild, and must
    refuse to measure anything the build half did not actually produce.
    """

    _RUNNER = os.path.join(
        os.path.dirname(os.path.dirname(store.__file__)), "_proxy_runner.py")

    def _run_phase(self, run_dir: str, phase: str):
        import subprocess
        env = {**os.environ,
               "MIMIR_PROXY_BENCH_DIR": store._CACHE_DIR,
               "MCP_FILES_ROOT": self.root}
        return subprocess.run(
            [sys.executable, self._RUNNER, "--run-dir", run_dir, "--phase", phase],
            capture_output=True, text=True, env=env, timeout=120)

    def _session(self, build_body: str = "echo built\n"):
        """A registered proxy whose build appends a line to a witness file."""
        self.witness = os.path.join(self.root, "build_ran")
        sh = _script(os.path.join(self.root, "b.sh"),
                     f"echo ran >> {self.witness}\n" + build_body)
        server_proxy.proxy_manage(
            op="register", name="tiny", executable_path=self._make_exe(),
            run_cmd_template="python3 {executable}",
            metadata={"build_cmd": sh}, confirm=True)
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "c0", "proxy_name": "tiny"}], confirm=True)
        run_dir = procs._new_run_dir(store._opt_session_runs_dir("tiny"))
        procs._write_run_config(run_dir, {
            "proxy_name": "tiny", "benchmark_name": "bench", "requirements": [],
        })
        return run_dir

    def _builds_ran(self) -> int:
        if not os.path.isfile(self.witness):
            return 0
        return len([ln for ln in open(self.witness) if ln.strip()])

    def test_the_build_phase_builds_and_stops_short_of_a_verdict(self) -> None:
        run_dir = self._session()
        res = self._run_phase(run_dir, "build")
        self.assertEqual(res.returncode, 0, msg=res.stdout + res.stderr)
        self.assertEqual(self._builds_ran(), 1)
        self.assertEqual(build.read_report(run_dir)["status"], "ok")
        # metrics.json is what makes a run read as finished, and this half measured
        # nothing — writing one here would report a run that never ran.
        self.assertFalse(os.path.isfile(os.path.join(run_dir, "metrics.json")))

    def test_the_run_phase_measures_without_building_again(self) -> None:
        run_dir = self._session()
        self._run_phase(run_dir, "build")
        res = self._run_phase(run_dir, "run")
        self.assertEqual(res.returncode, 0, msg=res.stdout + res.stderr)
        self.assertEqual(self._builds_ran(), 1)   # the build job's, not a second one
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "metrics.json")))

    def test_the_run_phase_refuses_a_build_that_failed(self) -> None:
        run_dir = self._session()
        build.write_report(run_dir, [{"proxy": "tiny", "status": "failed",
                                      "error": "build exited 2", "duration_s": 0.1}])
        res = self._run_phase(run_dir, "run")
        self.assertEqual(res.returncode, 2, msg=res.stdout + res.stderr)
        self.assertIn("Build failed", res.stdout)
        self.assertFalse(os.path.isfile(os.path.join(run_dir, "metrics.json")))

    def test_the_run_phase_refuses_a_build_that_never_happened(self) -> None:
        """A broken chain must not measure whatever binary was lying around."""
        run_dir = self._session()
        res = self._run_phase(run_dir, "run")
        self.assertEqual(res.returncode, 2, msg=res.stdout + res.stderr)
        self.assertIn("build job did not run", res.stdout)
        self.assertFalse(os.path.isfile(os.path.join(run_dir, "metrics.json")))



class RunProgressTests(_TmpStorageTest):
    """What a run says it is doing, read off the run dir while it is still going.

    A blocking call cannot report on itself — it does not answer until the run is
    over. Both halves of the answer are therefore read from disk: the phase from the
    sidecar the runner writes at its own boundaries, and the percentage from the
    build's own output, because the runner is blocked inside the build for its whole
    duration and is in no position to count it.
    """

    def setUp(self) -> None:
        super().setUp()
        self.run_dir = self.root

    def _write_phase(self, **fields) -> None:
        with open(procs._phase_path(self.run_dir), "w") as fh:
            json.dump(fields, fh)

    def _write_build_log(self, text: str) -> None:
        with open(procs._build_log_path(self.run_dir), "w") as fh:
            fh.write(text)

    # ── the build's own percentage ────────────────────────────────────────────
    def test_the_newest_percentage_wins(self) -> None:
        self._write_build_log("[  6%] Building A\n[ 50%] Building B\n[ 13%] Building C\n")
        # Not the largest: a suite of several builds restarts the count, and the run
        # is wherever the compiler last said it was.
        self.assertEqual(procs._build_percent(self.run_dir), 13.0)

    def test_padding_and_three_digits_are_read(self) -> None:
        self._write_build_log("[  7%] a\n")
        self.assertEqual(procs._build_percent(self.run_dir), 7.0)
        self._write_build_log("[100%] a\n")
        self.assertEqual(procs._build_percent(self.run_dir), 100.0)

    def test_a_build_that_counts_nothing_reports_nothing(self) -> None:
        # None is a real answer. Showing 0% for a build that never says how far
        # along it is would invent a fact about it.
        self._write_build_log("compiling everything, quietly\n")
        self.assertIsNone(procs._build_percent(self.run_dir))

    def test_an_absent_build_log_reports_nothing(self) -> None:
        self.assertIsNone(procs._build_percent(self.run_dir))

    def test_a_token_cut_in_half_by_the_tail_is_not_read(self) -> None:
        # The tail starts mid-line on any real build log. A partial "[ 1" must not
        # become a percentage.
        self._write_build_log("x" * 4096 + "[ 42%] done\n")
        self.assertEqual(procs._build_percent(self.run_dir, max_bytes=8), None)

    # ── the phase sidecar ─────────────────────────────────────────────────────
    def test_no_sidecar_means_the_run_says_nothing(self) -> None:
        self.assertEqual(procs._run_progress(self.run_dir), {})

    def test_a_corrupt_sidecar_is_silent(self) -> None:
        with open(procs._phase_path(self.run_dir), "w") as fh:
            fh.write("{not json")
        self.assertEqual(procs._run_progress(self.run_dir), {})

    def test_a_build_phase_carries_the_percentage(self) -> None:
        self._write_phase(kind="build", text="building tiny (1/2)")
        self._write_build_log("[ 34%] Building A\n")
        out = procs._run_progress(self.run_dir)
        self.assertEqual(out["phase"], "building tiny (1/2)")
        self.assertEqual(out["percent"], 34.0)

    def test_a_measurement_phase_carries_no_percentage(self) -> None:
        # The build log still holds the last thing the compiler said. Carrying it
        # into the measurement would leave a bar frozen at 98% for the rest of the
        # run, describing work that finished long ago.
        self._write_phase(kind="measure", text="case shock (2/3)")
        self._write_build_log("[ 98%] Building Z\n")
        out = procs._run_progress(self.run_dir)
        self.assertEqual(out["phase"], "case shock (2/3)")
        self.assertNotIn("percent", out)

    def test_the_phase_reaches_run_state(self) -> None:
        # One merge point, so the blocking wait, the status op and the detached
        # watcher all learn it at once.
        self._write_phase(kind="build", text="building tiny")
        self._write_build_log("[ 12%] a\n")
        st = procs._run_state(self.run_dir)
        self.assertEqual(st["phase"], "building tiny")
        self.assertEqual(st["percent"], 12.0)
        self.assertIn("state", st)


if __name__ == "__main__":
    unittest.main()
