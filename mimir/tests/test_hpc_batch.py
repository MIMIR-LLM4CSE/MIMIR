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
        self._orig_run = server_hpc._run_argv
        self._orig_dir = server_hpc._HPC_JOBS_DIR
        self._tmp = tempfile.TemporaryDirectory()
        server_hpc._HPC_JOBS_DIR = os.path.join(self._tmp.name, "jobs")

    def tearDown(self) -> None:
        server_hpc._run_argv = self._orig_run
        server_hpc._HPC_JOBS_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_requires_confirm(self) -> None:
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu")
        self.assertEqual(res.get("status"), "error")
        self.assertIn("confirm", res.get("hint", ""))

    def test_submits_and_returns_descriptor(self) -> None:
        server_hpc._run_argv = lambda argv, t: {
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
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": "no id here", "stderr": "", "returncode": 0}
        res = server_hpc.sbatch_submit(command="x", partition="cpu", confirm=True)
        self.assertEqual(res.get("status"), "error")
        self.assertIn("job id", res.get("error", ""))

    def test_invalid_walltime_rejected(self) -> None:
        res = server_hpc.sbatch_submit(command="x", partition="cpu",
                                       wall_time="notatime", confirm=True)
        self.assertEqual(res.get("status"), "error")


class SallocSubmitTests(unittest.TestCase):
    """The validation has to sit on the path that executes, not beside it.

    `salloc_submit` used to take a free-form command string and check only that it
    started with "salloc ", while the time/mem/shell-token checks lived in a separate
    build tool nothing forced the model to call. Resources are arguments now, so a
    rejected value can never reach the scheduler.
    """

    def setUp(self) -> None:
        self._orig = server_hpc._run_argv
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": "salloc: Granted job allocation 77",
            "stderr": "", "returncode": 0}

    def tearDown(self) -> None:
        server_hpc._run_argv = self._orig

    def test_unconfirmed_returns_the_exact_command_as_preview(self) -> None:
        res = server_hpc.salloc_submit(partition="cpu", nodes=2, mem="8G")
        self.assertEqual(res.get("status"), "error")
        self.assertIn("--partition=cpu", res["command"])
        self.assertIn("--nodes=2", res["command"])
        self.assertIn("--mem=8G", res["command"])

    def test_invalid_time_and_mem_are_rejected_on_the_executing_path(self) -> None:
        for kwargs in ({"time": "notatime"}, {"mem": "lots"}):
            res = server_hpc.salloc_submit(partition="cpu", confirm=True, **kwargs)
            self.assertEqual(res.get("status"), "error", kwargs)

    def test_partition_is_required(self) -> None:
        res = server_hpc.salloc_submit(partition="", confirm=True)
        self.assertEqual(res.get("status"), "error")

    def test_shell_metacharacters_stay_one_argv_token(self) -> None:
        # No shell is involved, so a metacharacter is just a bad partition name.
        res = server_hpc.salloc_submit(partition="cpu; rm -rf ~", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("'--partition=cpu; rm -rf ~'", res["command"])

    def test_extra_args_takes_flags_only(self) -> None:
        rejected = server_hpc.salloc_submit(partition="cpu", extra_args="--x=1 rm -rf /", confirm=True)
        self.assertEqual(rejected.get("status"), "error")
        accepted = server_hpc.salloc_submit(partition="cpu", extra_args="--exclusive", confirm=True)
        self.assertEqual(accepted.get("status"), "ok")
        self.assertIn("--exclusive", accepted["command"])


_SCONTROL = (
    "NodeName=gpu-n01 Arch=aarch64 CoresPerSocket=72 CPUAlloc=8 CPUEfctv=72 CPUTot=72 "
    "CPULoad=1.19 AvailableFeatures=gpu-node Gres=gpu:gh200:1(S:0) RealMemory=579000 "
    "AllocMem=4000 FreeMem=446444 Sockets=1 State=MIXED ThreadsPerCore=1 Partitions=gpu,gpu_night\n"
    "NodeName=cpu-n01 Arch=x86_64 CoresPerSocket=32 CPUAlloc=0 CPUEfctv=64 CPUTot=64 "
    "CPULoad=0.00 AvailableFeatures=(null) Gres=(null) RealMemory=773500 AllocMem=0 "
    "FreeMem=760000 Sockets=2 State=IDLE ThreadsPerCore=1 Partitions=cpu\n"
    "NodeName=cpu-n02 Arch=x86_64 CoresPerSocket=32 CPUAlloc=0 CPUEfctv=64 CPUTot=64 "
    "CPULoad=0.00 AvailableFeatures=(null) Gres=(null) RealMemory=773500 AllocMem=0 "
    "FreeMem=759000 Sockets=2 State=IDLE ThreadsPerCore=1 Partitions=cpu\n"
)


class SlurmNodesTests(unittest.TestCase):
    """Node inventory read from Slurm's own database — no allocation, so it can be
    consulted *before* choosing where to submit. Architecture is the field that earns
    the tool: on a mixed cluster a binary built on the login node does not run on an
    aarch64 compute node, and nothing else in the toolkit reports that."""

    def setUp(self) -> None:
        self._orig = server_hpc._run_argv
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": _SCONTROL, "stderr": "", "returncode": 0}

    def tearDown(self) -> None:
        server_hpc._run_argv = self._orig

    def test_aggregates_nodes_onto_hardware_types(self) -> None:
        res = server_hpc.slurm_nodes()
        self.assertEqual(res["nodes_total"], 3)
        self.assertEqual(res["type_count"], 2)
        self.assertEqual(res["architectures"], ["aarch64", "x86_64"])
        # Most immediately usable first: the two idle CPU nodes outrank the mixed GPU one.
        first = res["node_types"][0]
        self.assertEqual(first["arch"], "x86_64")
        self.assertEqual(first["nodes_total"], 2)
        self.assertEqual(first["by_state"], {"idle": 2})

    def test_reports_live_occupancy_and_gpu_type(self) -> None:
        node = server_hpc.slurm_nodes(node="gpu-n01")["nodes"][0]
        self.assertEqual(node["arch"], "aarch64")
        self.assertEqual(node["gres"], "gpu:gh200:1")   # socket affinity stripped
        self.assertEqual(node["cpus_allocated"], 8)
        self.assertEqual(node["cpus_free"], 64)
        self.assertEqual(node["mem_free_mb"], 446444)
        self.assertEqual(node["partitions"], ["gpu", "gpu_night"])
        self.assertEqual(node["features"], "gpu-node")

    def test_null_gres_and_features_become_empty(self) -> None:
        node = server_hpc.slurm_nodes(node="cpu-n01")["nodes"][0]
        self.assertEqual(node["gres"], "")
        self.assertEqual(node["features"], "")

    def test_filters_by_partition_and_state(self) -> None:
        self.assertEqual(server_hpc.slurm_nodes(partition="gpu")["nodes_total"], 1)
        self.assertEqual(server_hpc.slurm_nodes(states="idle")["nodes_total"], 2)
        self.assertEqual(server_hpc.slurm_nodes(partition="nope")["count"], 0)

    def test_falls_back_to_sinfo_without_scontrol(self) -> None:
        """A cluster that restricts scontrol still gets an answer — minus what only
        scontrol knows, and told so rather than defaulted."""
        server_hpc._run_argv = lambda argv, t: {"status": "error", "stderr": "denied"}
        rows = "cpu-n01|cpu|idle|64|773500|760000|(null)|2|32|1\ncpu-n01|cpu_night|idle|64|773500|760000|(null)|2|32|1\n"
        orig_bash = server_hpc._run_bash
        server_hpc._run_bash = lambda s, t: {"status": "ok", "stdout": rows, "stderr": "", "returncode": 0}
        try:
            res = server_hpc.slurm_nodes()
            self.assertEqual(res["nodes_total"], 1)   # one node in two partitions, not two nodes
            self.assertEqual(res["architectures"], [])
            self.assertIn("architecture", res["degraded"])
        finally:
            server_hpc._run_bash = orig_bash


if __name__ == "__main__":
    unittest.main()
