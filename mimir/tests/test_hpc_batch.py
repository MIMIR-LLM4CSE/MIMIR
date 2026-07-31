"""HPC async batch submission + normalized job status (background-jobs support).

The HPC server gains a non-blocking ``sbatch_submit`` (returns a job id + a
``background_job`` descriptor) and a ``slurm_job_status(job_id)`` shim that maps
squeue/sacct to the shared state vocabulary. Tests stub ``_run_bash`` so no real
scheduler is touched.

Run:
    python -m unittest mimir.tests.test_hpc_batch -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "hpc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_hpc  # noqa: E402


def _canned(mapping: dict):
    """Return a fake _run_bash whose output depends on the command substring."""
    def fake(script: str, timeout: int) -> dict:
        for key, (status, out) in mapping.items():
            if key in script:
                return {"status": status, "stdout": out, "stderr": "",
                        "returncode": 0 if status == "ok" else 1}
        return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}
    return fake


class NormalizedStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = server_hpc._run_bash

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig

    def _state(self, mapping: dict) -> str:
        server_hpc._run_bash = _canned(mapping)
        return server_hpc._normalized_job_state("123")[0]

    def test_running_and_pending_from_squeue(self) -> None:
        self.assertEqual(self._state({"squeue": ("ok", "RUNNING")}), "running")
        self.assertEqual(self._state({"squeue": ("ok", "PENDING")}), "pending")

    def test_done_from_sacct_when_out_of_queue(self) -> None:
        self.assertEqual(
            self._state({"squeue": ("ok", ""), "sacct": ("ok", "COMPLETED")}), "done")

    def test_crashed_from_sacct(self) -> None:
        self.assertEqual(
            self._state({"squeue": ("ok", ""), "sacct": ("ok", "FAILED")}), "crashed")
        self.assertEqual(
            self._state({"squeue": ("ok", ""), "sacct": ("ok", "TIMEOUT")}), "crashed")

    def test_unknown_when_nowhere(self) -> None:
        self.assertEqual(
            self._state({"squeue": ("ok", ""), "sacct": ("ok", "")}), "unknown")


class SlurmJobStatusToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = server_hpc._run_bash

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig

    def test_returns_normalized_state(self) -> None:
        server_hpc._run_bash = _canned({"squeue": ("ok", "RUNNING")})
        res = server_hpc.slurm_job_status(job_id="7")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["state"], "running")
        self.assertEqual(res["job_id"], "7")

    def test_empty_job_id_errors(self) -> None:
        self.assertEqual(server_hpc.slurm_job_status(job_id="").get("status"), "error")


class SbatchSubmitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_run = server_hpc._run_bash
        self._orig_dir = server_hpc._HPC_JOBS_DIR
        self._tmp = tempfile.TemporaryDirectory()
        server_hpc._HPC_JOBS_DIR = os.path.join(self._tmp.name, "jobs")

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig_run
        server_hpc._HPC_JOBS_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_requires_confirm(self) -> None:
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu")
        self.assertEqual(res.get("status"), "error")
        self.assertIn("confirm", res.get("hint", ""))

    def test_submits_and_returns_descriptor(self) -> None:
        server_hpc._run_bash = lambda s, t: {
            "status": "ok", "stdout": "Submitted batch job 4242",
            "stderr": "", "returncode": 0}
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res["job_id"], "4242")
        job = res["background_job"]
        self.assertEqual(job["server"], "hpc")
        self.assertEqual(job["job_key"], "4242")
        self.assertEqual(job["status_op"]["tool"], "slurm_job_status")
        self.assertEqual(job["status_op"]["args"]["job_id"], "4242")
        self.assertTrue(os.path.isfile(res["batch_script"]))
        with open(res["batch_script"]) as fh:
            script = fh.read()
        self.assertIn("#SBATCH --partition=cpu", script)
        self.assertIn("echo hi", script)

    def test_unparseable_sbatch_output_errors(self) -> None:
        server_hpc._run_bash = lambda s, t: {
            "status": "ok", "stdout": "no id here", "stderr": "", "returncode": 0}
        res = server_hpc.sbatch_submit(command="x", partition="cpu", confirm=True)
        self.assertEqual(res.get("status"), "error")
        self.assertIn("job id", res.get("error", ""))

    def test_invalid_walltime_rejected(self) -> None:
        res = server_hpc.sbatch_submit(command="x", partition="cpu",
                                       wall_time="notatime", confirm=True)
        self.assertEqual(res.get("status"), "error")


if __name__ == "__main__":
    unittest.main()
