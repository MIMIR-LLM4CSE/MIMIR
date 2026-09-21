"""HPC async batch submission + normalized job status (background-jobs support).

The HPC server gains a non-blocking ``sbatch_submit`` (returns a job id + a
``background_job`` descriptor) and a ``slurm_job_status(job_id)`` shim that maps
squeue/sacct to the shared state vocabulary. Tests stub ``_run_bash`` so no real
scheduler is touched.

Run:
    python -m unittest mimir.tests.test_hpc_batch -v
"""

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
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


class SbatchTargetingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_run = server_hpc._run_argv
        self._orig_dir = server_hpc._HPC_JOBS_DIR
        self._tmp = tempfile.TemporaryDirectory()
        server_hpc._HPC_JOBS_DIR = os.path.join(self._tmp.name, "jobs")
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": "Submitted batch job 7", "stderr": "", "returncode": 0}

    def tearDown(self) -> None:
        server_hpc._run_argv = self._orig_run
        server_hpc._HPC_JOBS_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_constraint_nodelist_exclusive_reach_the_script(self) -> None:
        res = server_hpc.sbatch_submit(command="./bench", partition="cpu", constraint="icelake",
                                       nodelist="n[01-02]", exclusive=True, ntasks=4,
                                       confirm=True)
        with open(res["batch_script"]) as fh:
            script = fh.read()
        for want in ("--constraint=icelake", "--nodelist=n[01-02]", "--exclusive", "--ntasks=4"):
            self.assertIn(want, script)
        self.assertNotIn("--nodes=", script)   # 0 = scheduler default

    def test_bad_constraint_rejected(self) -> None:
        res = server_hpc.sbatch_submit(command="x", partition="cpu", constraint="a b",
                                       confirm=True)
        self.assertEqual(res.get("status"), "error")


def _fake_scontrol(stdout: str):
    def fake(argv, t):
        if argv[:2] == ["scontrol", "show"]:
            return {"status": "ok", "stdout": stdout, "stderr": "", "returncode": 0}
        if argv[:1] == ["sbatch"]:
            return {"status": "ok", "stdout": "Submitted batch job 900", "stderr": "",
                    "returncode": 0}
        return {"status": "error", "stdout": "", "stderr": "", "returncode": 1}
    return fake


_NODE_LSCPU = ("Architecture: x86_64\nModel name: Intel(R) Xeon(R) Platinum 8358\n"
               "Socket(s): 2\nCore(s) per socket: 32\nThread(s) per core: 1\n"
               "L3 cache: 96 MiB (2 instances)\nFlags: avx2 fma avx512f avx512bw avx512vl\n")


class NodeProfileTests(unittest.TestCase):
    """Probing a compute node on the node, and reading the result back."""

    def setUp(self) -> None:
        self._saved = {k: getattr(server_hpc, k) for k in (
            "_run_argv", "_run_bash", "_HPC_JOBS_DIR", "_NODE_PROFILES_DIR",
            "_host_identity", "_local_profile_record")}
        self._tmp = tempfile.TemporaryDirectory()
        server_hpc._HPC_JOBS_DIR = os.path.join(self._tmp.name, "jobs")
        server_hpc._NODE_PROFILES_DIR = os.path.join(self._tmp.name, "profiles")
        server_hpc._run_argv = _fake_scontrol(_SCONTROL)
        server_hpc._run_bash = _canned({"squeue": ("ok", "PENDING")})
        # The host MIMIR runs on: a login node of another generation, on a newer OS.
        server_hpc._host_identity = lambda: {
            "arch": "x86_64", "os": {"id": "rhel", "version": "9.4", "glibc": "2.34"},
            "cpu": {"model": "Intel(R) Xeon(R) 6747P", "simd": {"avx2": True, "amx_tile": True}},
        }
        self._env = dict(os.environ)
        os.environ.pop("SLURM_JOB_ID", None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(server_hpc, k, v)
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    def _submit(self, **kw) -> str:
        res = server_hpc.slurm_probe_node(partition="cpu", confirm=True, **kw)
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res["background_job"]["status_op"]["args"]["job_id"], "900")
        return res["job_dir"]

    def _finish(self, job_dir: str, node: str, mode: str, **files) -> None:
        for name, text in {"node": node, "mode": mode, **files}.items():
            with open(os.path.join(job_dir, name), "w") as fh:
                fh.write(text)

    def test_probe_requires_confirm_and_valid_target(self) -> None:
        self.assertIn("confirm", server_hpc.slurm_probe_node(partition="cpu").get("hint", ""))
        bad = server_hpc.slurm_probe_node(partition="cpu", constraint="x;y", confirm=True)
        self.assertEqual(bad.get("status"), "error")

    def test_probe_runs_the_platform_profile_on_the_node(self) -> None:
        job_dir = self._submit(constraint="avx512")
        with open(os.path.join(job_dir, "probe.sh")) as fh:
            script = fh.read()
        self.assertIn("server_platform.py", script)
        self.assertIn("--profile-json", script)
        self.assertIn("--constraint=avx512", script)
        self.assertIn(".venv-", script)          # the node's own interpreter
        self.assertIn("lscpu", script)           # the no-Python fallback

    def test_unprobed_kind_says_so_and_pending_probe_is_listed(self) -> None:
        self._submit()
        res = server_hpc.slurm_node_profile(partition="cpu")
        kind = res["node_types"][0]
        self.assertFalse(kind["profiled"])
        self.assertEqual(res["pending_probes"][0]["job_id"], "900")
        self.assertEqual(res["execution_context"]["context"], "login")

    def test_full_probe_is_harvested_and_compared_with_the_host(self) -> None:
        job_dir = self._submit()
        profile = {
            "cpu": {"arch": "x86_64", "model": "Intel(R) Xeon(R) Platinum 8358",
                    "simd": {"avx2": True, "avx512f": True}},
            "os": {"id": "rhel", "version": "8.10", "glibc": "2.28"},
            "march": {"march": "icelake-server"},
            "gpu": {"available": False}, "toolchains": {"gcc": "gcc 8.5"},
            "conda_envs": {}, "virtualenvs": {},
        }
        self._finish(job_dir, "cpu-n01", "full", **{"profile.json": json.dumps(profile)})
        res = server_hpc.slurm_node_profile(partition="cpu")
        kind = res["node_types"][0]
        self.assertTrue(kind["profiled"])
        self.assertEqual(kind["profiled_on"], "cpu-n01")
        # Everything the host profile carries, not a thinner node view.
        self.assertEqual(kind["profile"]["toolchains"], {"gcc": "gcc 8.5"})
        self.assertEqual(kind["profile"]["march"]["march"], "icelake-server")
        match = kind["matches_this_host"]
        self.assertFalse(match["same"])
        self.assertEqual(set(match["differs"]), {"cpu_model", "simd", "os", "glibc"})
        # cpu-n02 shares Slurm's signature but was not read.
        self.assertEqual(kind["unprofiled_nodes"], 1)
        # Harvested once: a second read neither re-harvests nor loses it.
        again = server_hpc.slurm_node_profile(partition="cpu")
        self.assertTrue(again["node_types"][0]["profiled"])
        self.assertNotIn("pending_probes", again)

    def test_fallback_without_python_is_partial_but_parsed(self) -> None:
        job_dir = self._submit()
        fallback = ("==uname\nx86_64\n==lscpu\n" + _NODE_LSCPU +
                    "==march\n  -march=  \ticelake-server\n==gpu\n==os\nID=rhel\nVERSION_ID=8.10\n"
                    "==ldd\nldd (GNU libc) 2.28\n==nproc\n64\n")
        self._finish(job_dir, "cpu-n01", "partial", **{"fallback.txt": fallback})
        kind = server_hpc.slurm_node_profile(node="cpu-n01")["node_types"][0]
        self.assertIn("No MIMIR Python", kind["partial"])
        prof = kind["profile"]
        self.assertEqual(prof["cpu"]["model"], "Intel(R) Xeon(R) Platinum 8358")
        self.assertTrue(prof["cpu"]["simd"]["avx512f"])
        self.assertEqual(prof["march"]["march"], "icelake-server")
        self.assertEqual(prof["os"]["glibc"], "2.28")

    def test_profile_goes_stale_when_slurm_describes_the_node_differently(self) -> None:
        job_dir = self._submit()
        self._finish(job_dir, "cpu-n01", "full",
                     **{"profile.json": json.dumps({"cpu": {"model": "old"}})})
        server_hpc.slurm_node_profile(partition="cpu")
        # The node came back from maintenance with more memory: a different machine.
        server_hpc._run_argv = _fake_scontrol(_SCONTROL.replace("RealMemory=773500",
                                                                "RealMemory=1547000"))
        kind = server_hpc.slurm_node_profile(node="cpu-n01")["node_types"][0]
        self.assertFalse(kind["profiled"])
        self.assertIn("changed", kind["note"])

    def test_inside_an_allocation_the_host_is_the_node(self) -> None:
        """No job to submit: MIMIR already runs on the node, so its own profile is it."""
        os.environ.update({"SLURM_JOB_ID": "5", "SLURM_JOB_NODELIST": "cpu-n02"})
        calls = []

        def local(node):
            calls.append(node["node"])
            return {"node": node["node"], "probed_at": "now", "source": "this host",
                    "slurm": server_hpc._slurm_facts(node), "partial": False,
                    "profile": {"cpu": {"model": "Intel(R) Xeon(R) Platinum 8358"}}}

        server_hpc._local_profile_record = local
        with unittest.mock.patch.object(server_hpc.socket, "gethostname", lambda: "cpu-n02"):
            res = server_hpc.slurm_node_profile(partition="cpu")
        self.assertEqual(res["execution_context"]["context"], "in_allocation")
        self.assertEqual(calls, ["cpu-n02"])
        self.assertTrue(res["node_types"][0]["profiled"])
        self.assertFalse(os.listdir(server_hpc._HPC_JOBS_DIR) if os.path.isdir(
            server_hpc._HPC_JOBS_DIR) else [])


if __name__ == "__main__":
    unittest.main()
