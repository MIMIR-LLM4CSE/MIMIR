"""Slurm submission for the proxy server (_ops/slurm.py, _lib/placement.py).

Resource validation, the sbatch script actually handed to the scheduler, and the
build/run split: one job by default, two chained jobs when the build is sent to a
partition of its own. Gathered here because these were spread across the helper and
op test modules, neither of which is about submission.

Every test runs against a fake ``sbatch`` on a PATH with nothing else on it, so no
scheduler can be reached even by accident.

Run:
    python -m unittest mimir.tests.test_proxy_slurm -v
"""

import os
import unittest
from unittest import mock

from mimir.tests._proxy_fixtures import (  # noqa: F401
    _FakeSbatchTest, _TmpStorageTest, build, procs, server_proxy, store,
)


class ValidateSlurmArgsTests(unittest.TestCase):
    def test_valid_args_pass(self) -> None:
        self.assertIsNone(procs._validate_slurm_args("gpu", 1, 8, "32G", "04:00:00"))
        self.assertIsNone(procs._validate_slurm_args("cpu", 0, 1, "64000M", "1-12:00:00"))

    def test_invalid_args_rejected(self) -> None:
        self.assertIn("partition", procs._validate_slurm_args("", 0, 8, "32G", "04:00:00"))
        self.assertIn("gpus", procs._validate_slurm_args("p", -1, 8, "32G", "04:00:00"))
        self.assertIn("mem", procs._validate_slurm_args("p", 0, 8, "lots", "04:00:00"))
        self.assertIn("wall_time", procs._validate_slurm_args("p", 0, 8, "32G", "4 hours"))


class ValidateSlurmTokenTests(unittest.TestCase):
    """constraint/nodelist land in #SBATCH lines, so a newline is an injection."""

    def test_real_feature_expressions_pass(self) -> None:
        for value in ("", "a100", "bigmem&avx512", "gpu|cpu", "node[01-04]",
                      "(a100|v100)&nvlink", "skylake*2"):
            self.assertIsNone(procs._validate_slurm_token(value, "constraint"), value)

    def test_a_newline_cannot_smuggle_in_another_directive(self) -> None:
        bad = procs._validate_slurm_token(
            "a100\n#SBATCH --partition=everything", "constraint")
        self.assertIsNotNone(bad)
        self.assertIn("constraint", bad)

    def test_shell_punctuation_is_refused(self) -> None:
        for value in ("a100;rm -rf /", "$(hostname)", "a 100", "a100`id`"):
            self.assertIsNotNone(procs._validate_slurm_token(value, "nodelist"), value)


class ValidateSlurmCommentTests(unittest.TestCase):
    """A comment is free text, so only length and control characters are refused."""

    def test_real_comments_pass(self) -> None:
        for value in ("", "ratchet iter 12", "GEOS-X / seismic #4 (baseline)",
                      "why: comparing against ref_2024 — 50% mesh"):
            self.assertIsNone(procs._validate_slurm_comment(value), value)

    def test_a_newline_cannot_smuggle_in_another_directive(self) -> None:
        bad = procs._validate_slurm_comment(
            "baseline\n#SBATCH --partition=everything")
        self.assertIsNotNone(bad)
        self.assertIn("comment", bad)

    def test_other_control_characters_are_refused(self) -> None:
        for value in ("a\tb", "a\rb", "a\x00b", "a\x7fb"):
            self.assertIsNotNone(procs._validate_slurm_comment(value), value)

    def test_an_overlong_comment_is_refused(self) -> None:
        bad = procs._validate_slurm_comment("x" * 513)
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
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(d, "#!/bin/bash\ntrue\n")
        self.assertIsNone(error)
        self.assertEqual(job_id, 4242)
        self.assertEqual(procs._read_slurm_id(d), 4242)
        self.assertTrue(os.path.isfile(os.path.join(d, "batch_script.sh")))

    def test_a_named_script_and_id_file_keep_two_jobs_apart(self) -> None:
        """A split build/run submission puts two jobs in one run directory."""
        self._fake_sbatch("#!/bin/sh\necho 'Submitted batch job 4243'\n")
        os.environ["PATH"] = self.bin_dir
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(
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
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(d, "#!/bin/bash\ntrue\n")
        self.assertIsNone(job_id)
        self.assertIn("sbatch failed", error["error"])

    def test_sbatch_missing_suggests_local_alternative(self) -> None:
        os.environ["PATH"] = self.bin_dir  # empty dir: no sbatch
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(
            d, "#!/bin/bash\ntrue\n",
            local_alternative="proxy_exec(op='run', confirm=True)",
        )
        self.assertIsNone(job_id)
        self.assertIn("sbatch not found", error["error"])
        self.assertIn("proxy_exec(op='run'", error["hint"])


class SubmitEvalTests(_FakeSbatchTest):
    """proxy_slurm(op='eval') must not lose what _prepare_run wrote."""

    def test_submitted_run_keeps_the_convergence_config(self) -> None:
        self._session(convergence={"h_param": "n", "error_metric": "l2_rel"})
        res = server_proxy.proxy_slurm(op="eval", partition="debug", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)

        cfg = store._read_json(os.path.join(res["run_dir"], "config.json"))
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

        cfg = store._read_json(os.path.join(run_dir, "config.json"))
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
        # squeue is not on PATH here, so the job reads as 'unknown'; force the state
        # the guard is about and assert the build report alone settles it.
        with mock.patch.object(procs, "_squeue_state", return_value="pending"):
            self.assertEqual(procs._run_state(run_dir)["state"], "pending")
            build.write_report(run_dir, [{"proxy": "tiny", "status": "failed",
                                          "error": "build exited 2", "duration_s": 1.0}])
            self.assertEqual(procs._run_state(run_dir)["state"], "crashed")

    def test_comment_reaches_both_jobs_of_a_split_run(self) -> None:
        """The comment labels the submission, so either job answers 'what was this'."""
        self._session(metadata={"build_cmd": "make all", "build_partition": "compile"})
        res = server_proxy.proxy_slurm(
            op="eval", partition="gpu_simu", comment="ratchet iter 12", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        build_script, run_script = self._submitted_scripts()
        for script in (build_script, run_script):
            self.assertIn("#SBATCH --comment='ratchet iter 12'", script)
        cfg = store._read_json(os.path.join(res["run_dir"], "config.json"))
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
        """_validate_slurm_token covers the refusal; this covers its timing."""
        self._session()
        res = server_proxy.proxy_slurm(
            op="eval", partition="debug",
            constraint="a100\n#SBATCH --partition=everything", confirm=True)
        self.assertEqual(res.get("status"), "error", msg=res)
        self.assertEqual(self._submitted_scripts(), [])

if __name__ == "__main__":
    unittest.main()
