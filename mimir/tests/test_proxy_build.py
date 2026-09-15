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
import time
import unittest

from mimir.tests.test_proxy_ops import (  # noqa: F401
    _TmpStorageTest, _eval, eval_session, ratchet, registry, server_proxy, store,
)

from _lib import build, procs, tree_snapshot  # noqa: E402


def _script(path: str, body: str) -> str:
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n" + body)
    os.chmod(path, 0o755)
    return path


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
        self.assertEqual(sp["timeout_s"], 1800.0)
        huge = build.spec({"build_cmd": "make", "build_timeout_s": 10 ** 9})
        self.assertLessEqual(huge["timeout_s"], 6 * 3600.0)


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
        eval_session._save_opt_config({
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
        cfg = eval_session._load_opt_config("tiny")
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

        self.assertTrue(tree_snapshot._copy_restore(git_dir, self.root, [src], snap))
        self.assertEqual(open(src).read(), "int best(){return 1;}\n")
        self.assertGreater(os.path.getmtime(src), artifact_built_at)


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


if __name__ == "__main__":
    unittest.main()
