"""Slurm submission for the proxy server: resource validation, the sbatch script, and the
build/run split. A fake ``sbatch`` on a bare PATH; no scheduler is reachable.
"""

from mimir.servers.proxy._lib import layout
import json
import os
import unittest
from unittest import mock

from mimir.tests.support.proxy import (
    _FakeSbatchTest,
    _TmpStorageTest,
    build,
    procs,
    server_proxy,
    store,
)
from mimir.shared import jobs as shared_jobs, slurm_site


class ValidateSlurmArgsTests(unittest.TestCase):
    def test_valid_args_pass(self) -> None:
        self.assertIsNone(procs.validate_slurm_args("gpu", 1, 8, "32G", "04:00:00"))
        self.assertIsNone(procs.validate_slurm_args("cpu", 0, 1, "64000M", "1-12:00:00"))

    def test_invalid_args_rejected(self) -> None:
        self.assertIn("partition", procs.validate_slurm_args("", 0, 8, "32G", "04:00:00"))
        self.assertIn("gpus", procs.validate_slurm_args("p", -1, 8, "32G", "04:00:00"))
        self.assertIn("mem", procs.validate_slurm_args("p", 0, 8, "lots", "04:00:00"))
        self.assertIn("wall_time", procs.validate_slurm_args("p", 0, 8, "32G", "4 hours"))


class ValidateSlurmTokenTests(unittest.TestCase):
    """constraint/nodelist land in #SBATCH lines, so a newline is an injection."""

    def test_real_feature_expressions_pass(self) -> None:
        for value in ("", "a100", "bigmem&avx512", "gpu|cpu", "node[01-04]",
                      "(a100|v100)&nvlink", "skylake*2"):
            self.assertIsNone(procs.validate_slurm_token(value, "constraint"), value)

    def test_a_newline_cannot_smuggle_in_another_directive(self) -> None:
        bad = procs.validate_slurm_token(
            "a100\n#SBATCH --partition=everything", "constraint")
        self.assertIsNotNone(bad)
        self.assertIn("constraint", bad)

    def test_shell_punctuation_is_refused(self) -> None:
        for value in ("a100;rm -rf /", "$(hostname)", "a 100", "a100`id`"):
            self.assertIsNotNone(procs.validate_slurm_token(value, "nodelist"), value)


class ValidateSlurmCommentTests(unittest.TestCase):
    """A comment is free text, so only length and control characters are refused."""

    def test_real_comments_pass(self) -> None:
        for value in ("", "ratchet iter 12", "GEOS-X / seismic #4 (baseline)",
                      "why: comparing against ref_2024 — 50% mesh"):
            self.assertIsNone(procs.validate_slurm_comment(value), value)

    def test_a_newline_cannot_smuggle_in_another_directive(self) -> None:
        bad = procs.validate_slurm_comment(
            "baseline\n#SBATCH --partition=everything")
        self.assertIsNotNone(bad)
        self.assertIn("comment", bad)

    def test_other_control_characters_are_refused(self) -> None:
        for value in ("a\tb", "a\rb", "a\x00b", "a\x7fb"):
            self.assertIsNotNone(procs.validate_slurm_comment(value), value)

    def test_an_overlong_comment_is_refused(self) -> None:
        bad = procs.validate_slurm_comment("x" * 513)
        self.assertIsNotNone(bad)
        self.assertIn("too long", bad)


class SubmitSbatchTests(_TmpStorageTest):
    """The submission primitive itself, against a one-shot fake `sbatch` on PATH.

    Separate from _FakeSbatchTest, whose fake hands out incrementing ids: these
    tests need to control exactly what sbatch prints, including failing.
    """

    def setUp(self) -> None:
        super().setUp()
        self.bin_dir = os.path.join(self.root, "bin")
        os.makedirs(self.bin_dir, exist_ok=True)
        self._saved_path = os.environ["PATH"]

    def tearDown(self) -> None:
        os.environ["PATH"] = self._saved_path
        super().tearDown()

    def _fake_sbatch(self, script: str) -> None:
        p = os.path.join(self.bin_dir, "sbatch")
        with open(p, "w") as fh:
            fh.write(script)
        os.chmod(p, 0o755)

    def test_success_parses_and_records_job_id(self) -> None:
        self._fake_sbatch("#!/bin/sh\necho 'Submitted batch job 4242'\n")
        os.environ["PATH"] = self.bin_dir
        d = procs.new_run_dir(self.root)
        job_id, error = procs.submit_sbatch(d, "#!/bin/bash\ntrue\n")
        self.assertIsNone(error)
        self.assertEqual(job_id, 4242)
        self.assertEqual(procs._read_slurm_id(d), 4242)
        self.assertTrue(os.path.isfile(os.path.join(d, "batch_script.sh")))

    def test_a_named_script_and_id_file_keep_two_jobs_apart(self) -> None:
        """A split build/run submission puts two jobs in one run directory."""
        self._fake_sbatch("#!/bin/sh\necho 'Submitted batch job 4243'\n")
        os.environ["PATH"] = self.bin_dir
        d = procs.new_run_dir(self.root)
        job_id, error = procs.submit_sbatch(
            d, "#!/bin/bash\ntrue\n",
            script_name="build_batch_script.sh", id_file="build_slurm_job_id")
        self.assertIsNone(error)
        self.assertEqual(procs._read_build_slurm_id(d), 4243)
        self.assertTrue(os.path.isfile(os.path.join(d, "build_batch_script.sh")))
        # The run job's own slots are untouched, so the watcher still has nothing
        # to follow until the run job is submitted.
        self.assertIsNone(procs._read_slurm_id(d))
        self.assertFalse(os.path.isfile(os.path.join(d, "batch_script.sh")))

    def test_sbatch_failure_returns_err(self) -> None:
        self._fake_sbatch("#!/bin/sh\necho 'bad partition' >&2\nexit 1\n")
        os.environ["PATH"] = self.bin_dir
        d = procs.new_run_dir(self.root)
        job_id, error = procs.submit_sbatch(d, "#!/bin/bash\ntrue\n")
        self.assertIsNone(job_id)
        self.assertIn("sbatch failed", error["error"])

    def test_sbatch_missing_suggests_local_alternative(self) -> None:
        os.environ["PATH"] = self.bin_dir  # empty dir: no sbatch
        d = procs.new_run_dir(self.root)
        job_id, error = procs.submit_sbatch(
            d, "#!/bin/bash\ntrue\n",
            local_alternative="proxy_exec(op='run', confirm=True)",
        )
        self.assertIsNone(job_id)
        self.assertIn("sbatch not found", error["error"])
        self.assertIn("proxy_exec(op='run'", error["hint"])


class SubmitEvalTests(_FakeSbatchTest):
    """proxy_slurm(op='eval') must not lose what prepare_run wrote."""

    def test_submitted_run_keeps_the_convergence_config(self) -> None:
        self._session(convergence={"h_param": "n", "error_metric": "l2_rel"})
        res = server_proxy.proxy_slurm(op="eval", partition="debug", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)

        cfg = layout.read_json(os.path.join(res["run_dir"], "config.json"))
        # The Slurm-specific key is merged in, not written over the rest.
        self.assertEqual(cfg["partition"], "debug")
        self.assertEqual(cfg["convergence"], {"h_param": "n", "error_metric": "l2_rel"})
        self.assertEqual(cfg["benchmark_name"], "bench")
        self.assertEqual(cfg["deadline_s"], 3.0 * 3600)

    def test_without_a_build_partition_it_is_still_one_job(self) -> None:
        """The non-regression that matters: nothing splits unless asked."""
        self._session(metadata={"build_cmd": "make all"})
        res = server_proxy.proxy_slurm(op="eval", partition="debug", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)

        scripts = self._submitted_scripts()
        self.assertEqual(len(scripts), 1)
        self.assertNotIn("--phase", scripts[0])
        self.assertNotIn("--dependency", scripts[0])
        self.assertNotIn("build_job_id", res)
        self.assertFalse(os.path.isfile(
            os.path.join(res["run_dir"], "build_slurm_job_id")))

    def test_build_partition_splits_into_two_chained_jobs(self) -> None:
        self._session(metadata={"build_cmd": "make all"})
        res = server_proxy.proxy_slurm(
            op="eval", partition="gpu_simu", constraint="a100",
            build_partition="compile", build_cpus_per_task=32, confirm=True,
        )
        self.assertEqual(res.get("status"), "ok", msg=res)

        build_script, run_script = self._submitted_scripts()
        # Each half went where it was told, and the build did not inherit the GPU
        # node's feature expression — that is the whole point of the split.
        self.assertIn("#SBATCH --partition=compile", build_script)
        self.assertIn("#SBATCH --cpus-per-task=32", build_script)
        self.assertNotIn("--constraint", build_script)
        self.assertIn("--phase build", build_script)

        self.assertIn("#SBATCH --partition=gpu_simu", run_script)
        self.assertIn("#SBATCH --constraint=a100", run_script)
        self.assertIn("--phase run", run_script)
        # Chained, and a build that fails kills the run rather than queueing it forever.
        self.assertIn(f"#SBATCH --dependency=afterok:{res['build_job_id']}", run_script)
        self.assertIn("#SBATCH --kill-on-invalid-dep=yes", run_script)

        # slurm_job_id stays the RUN job: it is the one whose end is the run's end,
        # and every watcher follows it.
        run_dir = res["run_dir"]
        self.assertEqual(procs._read_slurm_id(run_dir), res["slurm_job_id"])
        self.assertEqual(procs._read_build_slurm_id(run_dir), res["build_job_id"])
        self.assertNotEqual(res["slurm_job_id"], res["build_job_id"])
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "build_batch_script.sh")))

        cfg = layout.read_json(os.path.join(run_dir, "config.json"))
        self.assertEqual(cfg["placement"]["build"]["partition"], "compile")
        self.assertEqual(cfg["placement"]["run"]["constraint"], "a100")

    def test_registered_build_partition_is_the_standing_default(self) -> None:
        """The ratchet submits the same run hundreds of times; it is declared once."""
        self._session(metadata={"build_cmd": "make all", "build_partition": "compile",
                                "build_mem": "128G"})
        res = server_proxy.proxy_slurm(op="eval", partition="gpu_simu", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        build_script, _run = self._submitted_scripts()
        self.assertIn("#SBATCH --partition=compile", build_script)
        self.assertIn("#SBATCH --mem=128G", build_script)
        self.assertEqual(res["build_partition"], "compile")

    def test_a_proxy_with_no_build_is_never_split(self) -> None:
        """Nothing to build means nothing to place: one job, whatever was asked."""
        self._session()  # no build_cmd
        res = server_proxy.proxy_slurm(
            op="eval", partition="gpu_simu", build_partition="compile", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        self.assertEqual(len(self._submitted_scripts()), 1)
        self.assertNotIn("build_job_id", res)

    def test_a_failed_build_makes_the_pending_run_read_as_crashed(self) -> None:
        """Otherwise the run sits PENDING until Slurm gets around to it."""
        self._session(metadata={"build_cmd": "make all", "build_partition": "compile"})
        res = server_proxy.proxy_slurm(op="eval", partition="gpu_simu", confirm=True)
        run_dir = res["run_dir"]
        # squeue is not on PATH here; force the state the guard is about and assert
        # the build report alone settles it.
        with mock.patch.object(procs, "_slurm_job_state", return_value=("pending", "", [])):
            self.assertEqual(procs.run_state(run_dir)["state"], "pending")
            build.write_report(run_dir, [{"proxy": "tiny", "status": "failed",
                                          "error": "build exited 2", "duration_s": 1.0}])
            self.assertEqual(procs.run_state(run_dir)["state"], "crashed")

    def test_comment_reaches_both_jobs_of_a_split_run(self) -> None:
        """The comment labels the submission, so either job answers 'what was this'."""
        self._session(metadata={"build_cmd": "make all", "build_partition": "compile"})
        res = server_proxy.proxy_slurm(
            op="eval", partition="gpu_simu", comment="ratchet iter 12", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        build_script, run_script = self._submitted_scripts()
        for script in (build_script, run_script):
            self.assertIn("#SBATCH --comment='ratchet iter 12'", script)
        cfg = layout.read_json(os.path.join(res["run_dir"], "config.json"))
        self.assertEqual(cfg["placement"]["run"]["comment"], "ratchet iter 12")
        self.assertEqual(cfg["placement"]["build"]["comment"], "ratchet iter 12")

    def test_no_comment_leaves_the_directive_out(self) -> None:
        self._session()
        res = server_proxy.proxy_slurm(op="eval", partition="debug", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        self.assertNotIn("--comment", self._submitted_scripts()[0])

    def test_a_refused_comment_submits_nothing(self) -> None:
        self._session()
        res = server_proxy.proxy_slurm(
            op="eval", partition="debug",
            comment="baseline\n#SBATCH --partition=everything", confirm=True)
        self.assertEqual(res.get("status"), "error", msg=res)
        self.assertEqual(self._submitted_scripts(), [])

    def test_a_refused_argument_submits_nothing(self) -> None:
        """validate_slurm_token covers the refusal; this covers its timing."""
        self._session()
        res = server_proxy.proxy_slurm(
            op="eval", partition="debug",
            constraint="a100\n#SBATCH --partition=everything", confirm=True)
        self.assertEqual(res.get("status"), "error", msg=res)
        self.assertEqual(self._submitted_scripts(), [])

if __name__ == "__main__":
    unittest.main()


class ARunIsAJobOfTheWorkspaceTests(_FakeSbatchTest):
    """A proxy run submitted to Slurm is also filed as a workspace job descriptor, so nothing
    offers to watch it twice.
    """

    def _submitted(self, **kwargs) -> tuple[str, int]:
        run_dir = os.path.join(store.runs_dir(), "run-0001")
        os.makedirs(run_dir, exist_ok=True)
        job_id, failure = procs.submit_sbatch(run_dir, "#!/bin/bash\necho hi\n",
                                               **kwargs)
        self.assertIsNone(failure)
        return run_dir, job_id

    def test_the_run_is_filed_where_the_workspace_accounts_for_its_jobs(self) -> None:
        _, job_id = self._submitted()
        self.assertIn(("", str(job_id)), slurm_site.recorded_jobs())

    def test_so_it_is_not_named_as_work_nobody_is_watching(self) -> None:
        _, job_id = self._submitted()
        self.assertEqual(slurm_site.unwatched([str(job_id)], host=""), [])

    def test_both_halves_of_a_split_submission_are_accounted_for(self) -> None:
        """A build job sits in the same queue and is just as much somebody's row."""
        run_dir, run_id = self._submitted()
        build_id, failure = procs.submit_sbatch(
            run_dir, "#!/bin/bash\nmake\n",
            script_name="build_batch.sh", id_file="build_slurm_job_id")
        self.assertIsNone(failure)
        recorded = {job for _, job in slurm_site.recorded_jobs()}
        self.assertEqual({str(run_id), str(build_id)} - recorded, set())

    def test_the_descriptor_names_the_watcher_this_server_runs(self) -> None:
        """So a reload re-arms *this* watcher rather than a stock poll beside it."""
        run_dir, _ = self._submitted()
        procs.background_descriptor(
            run_dir, kind="proxy-eval",
            status_op={"tool": "proxy_eval_status", "args": {"proxy_name": "solver"}},
            summary_op={})
        job_dir = procs._workspace_job_dir(run_dir)
        with open(os.path.join(job_dir, shared_jobs.WATCHER_FILE)) as fh:
            watcher = json.load(fh)
        self.assertEqual(watcher["server"], "proxy")
        self.assertEqual(watcher["job_key"], "run-0001")
        self.assertEqual(watcher["status_op"]["tool"], "proxy_eval_status")

    def test_a_finished_run_is_settled_in_the_workspace_job_dir(self) -> None:
        """Otherwise the workspace scan reads it as live, and the server never idles."""
        run_dir, _ = self._submitted()
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            fh.write("{}")
        with mock.patch.object(procs, "_slurm_job_state", return_value=("done", "", [])):
            self.assertEqual(procs.run_state(run_dir)["state"], "done")
        self.assertEqual(
            shared_jobs.recorded_terminal_state(procs._workspace_job_dir(run_dir)), "done")

    def test_only_the_run_job_carries_the_watcher(self) -> None:
        """Both jobs are accounted for; only the run's end is the run's end.

        A watcher on the build job too would wake the conversation when the build
        finished, which is the middle of the run rather than its result.
        """
        run_dir, run_id = self._submitted()
        procs.submit_sbatch(run_dir, "#!/bin/bash\nmake\n",
                             script_name="build_batch.sh",
                             id_file="build_slurm_job_id")
        procs.background_descriptor(
            run_dir, kind="proxy-eval",
            status_op={"tool": "proxy_eval_status", "args": {"proxy_name": "solver"}},
            summary_op={})
        dirs = [os.path.join(slurm_site.jobs_dir(), n)
                for n in os.listdir(slurm_site.jobs_dir())]
        carried = [d for d in dirs
                   if os.path.exists(os.path.join(d, shared_jobs.WATCHER_FILE))]
        self.assertEqual(len(dirs), 2)
        self.assertEqual(carried, [procs._workspace_job_dir(run_dir)])
        with open(os.path.join(carried[0], "slurm_job_id")) as fh:
            self.assertEqual(fh.read().strip(), str(run_id))
