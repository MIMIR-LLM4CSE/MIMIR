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
from unittest.mock import patch
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
        # Job dirs land under the submitting session's state dir; point the state dir at
        # a temp tree. Through the environment, so the test exercises the real
        # resolution (state_paths.session_state_dir) rather than a patched constant.
        self._orig_run = server_hpc._run_argv
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._state = os.path.join(self._tmp.name, "state")
        os.makedirs(self._state)
        env = patch.dict(os.environ, {"MIMIR_STATE_DIR": self._state}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MIMIR_SESSION_ID", None)
        os.environ.pop("MIMIR_HPC_JOBS_DIR", None)

    def tearDown(self) -> None:
        server_hpc._run_argv = self._orig_run

    def _submit(self, **kwargs) -> dict:
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": "Submitted batch job 4242",
            "stderr": "", "returncode": 0}
        return server_hpc.sbatch_submit(
            command=kwargs.pop("command", "echo hi"),
            partition=kwargs.pop("partition", "cpu"),
            confirm=True, **kwargs)

    def test_two_submissions_in_the_same_second_get_two_directories(self) -> None:
        """The stamp resolves to the second, and exist_ok=True shared the directory.

        Two sessions submitting at the same moment overwrote each other's
        batch_script.sh, slurm.log and slurm_job_id.
        """
        first = self._submit()["job_dir"]
        second = self._submit()["job_dir"]
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.isfile(os.path.join(first, "batch_script.sh")))
        self.assertTrue(os.path.isfile(os.path.join(second, "batch_script.sh")))

    def test_each_session_submits_into_its_own_directory(self) -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            mine = self._submit()["job_dir"]
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-b"}):
            theirs = self._submit()["job_dir"]
        self.assertIn(os.path.join("sessions", "session-a", "hpc_jobs"), mine)
        self.assertIn(os.path.join("sessions", "session-b", "hpc_jobs"), theirs)

    def test_a_job_submitted_elsewhere_is_still_findable_and_says_so(self) -> None:
        """A Slurm job outlives its conversation; filing it per session must not hide it."""
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            submitted = self._submit()["job_dir"]
        self.addCleanup(setattr, server_hpc, "_normalized_job_state",
                        server_hpc._normalized_job_state)
        server_hpc._normalized_job_state = lambda jid: ("running", "RUNNING")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-b"}):
            status = server_hpc.slurm_job_status(job_id="4242")
        self.assertEqual(status["job_dir"], submitted)
        self.assertEqual(status["submitted_by_another_session"], "session-a")

    def test_a_job_of_this_session_is_not_labelled_as_another_s(self) -> None:
        self.addCleanup(setattr, server_hpc, "_normalized_job_state",
                        server_hpc._normalized_job_state)
        server_hpc._normalized_job_state = lambda jid: ("running", "RUNNING")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            submitted = self._submit()["job_dir"]
            status = server_hpc.slurm_job_status(job_id="4242")
        self.assertEqual(status["job_dir"], submitted)
        self.assertNotIn("submitted_by_another_session", status)

    def _status_as(self, state: str, job_id: str = "4242") -> dict:
        self.addCleanup(setattr, server_hpc, "_normalized_job_state",
                        server_hpc._normalized_job_state)
        server_hpc._normalized_job_state = lambda jid: (state, state.upper())
        return server_hpc.slurm_job_status(job_id=job_id)

    def test_a_terminal_poll_settles_the_submission_directory(self) -> None:
        """The counterpart of a shell job's exit-code trap.

        A submission records an id, and an id never stops existing — so whoever asks
        later whether this workspace still has work running reads the directory as live
        for ever. This poll is the only thing that will ever know otherwise.
        """
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            job_dir = self._submit()["job_dir"]
            self._status_as("done")
        with open(os.path.join(job_dir, "slurm_state")) as fh:
            self.assertEqual(fh.read().strip(), "done")

    def test_a_job_still_in_the_queue_is_not_settled(self) -> None:
        for state in ("running", "pending"):
            with self.subTest(state=state), \
                 patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
                job_dir = self._submit()["job_dir"]
                self._status_as(state)
                self.assertFalse(os.path.exists(os.path.join(job_dir, "slurm_state")))

    def test_a_job_slurm_has_forgotten_is_settled_as_unknown(self) -> None:
        # Terminal and claiming nothing. Leaving it blank would read as "still running"
        # for ever, which is the failure this whole record exists to end.
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            job_dir = self._submit()["job_dir"]
            self._status_as("unknown")
        with open(os.path.join(job_dir, "slurm_state")) as fh:
            self.assertEqual(fh.read().strip(), "unknown")

    def test_the_first_observation_stands(self) -> None:
        # sacct's retention window expires and the same job then reads 'unknown'.
        # Overwriting would turn every recorded outcome into "forgotten" given time.
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            job_dir = self._submit()["job_dir"]
            self._status_as("crashed")
            server_hpc._normalized_job_state = lambda jid: ("unknown", "")
            server_hpc.slurm_job_status(job_id="4242")
        with open(os.path.join(job_dir, "slurm_state")) as fh:
            self.assertEqual(fh.read().strip(), "crashed")

    def test_another_session_s_poll_settles_it_too(self) -> None:
        # The record belongs to the job, not to whoever looked — and the reader that
        # needs it most is scanning sessions that may have no agent at all.
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            job_dir = self._submit()["job_dir"]
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-b"}):
            self._status_as("done")
        self.assertTrue(os.path.exists(os.path.join(job_dir, "slurm_state")))

    def test_an_unwritable_directory_does_not_cost_the_answer(self) -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            job_dir = self._submit()["job_dir"]
            os.chmod(job_dir, 0o500)
            self.addCleanup(os.chmod, job_dir, 0o700)
            status = self._status_as("done")
        self.assertEqual(status["state"], "done")

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

    def test_comment_lands_in_the_script_quoted(self) -> None:
        """Free text with spaces still has to be one --comment value to Slurm."""
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": "Submitted batch job 7", "stderr": "",
            "returncode": 0}
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu",
                                       comment="mimir ratchet iter 12", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        self.assertEqual(res["comment"], "mimir ratchet iter 12")
        with open(res["batch_script"]) as fh:
            self.assertIn("#SBATCH --comment='mimir ratchet iter 12'", fh.read())

    def test_without_a_comment_the_directive_is_absent(self) -> None:
        server_hpc._run_argv = lambda argv, t: {
            "status": "ok", "stdout": "Submitted batch job 8", "stderr": "",
            "returncode": 0}
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu", confirm=True)
        with open(res["batch_script"]) as fh:
            self.assertNotIn("--comment", fh.read())

    def test_a_newline_in_the_comment_is_refused_before_slurm(self) -> None:
        """It would otherwise forge a #SBATCH directive line of its own."""
        called: list = []
        server_hpc._run_argv = lambda argv, t: called.append(argv) or {
            "status": "ok", "stdout": "Submitted batch job 9", "stderr": "",
            "returncode": 0}
        res = server_hpc.sbatch_submit(
            command="echo hi", partition="cpu", confirm=True,
            comment="baseline\n#SBATCH --partition=everything")
        self.assertEqual(res.get("status"), "error", msg=res)
        self.assertEqual(called, [])


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

    def test_comment_is_one_argv_token_and_validated(self) -> None:
        accepted = server_hpc.salloc_submit(partition="cpu", comment="debug session",
                                            confirm=True)
        self.assertEqual(accepted.get("status"), "ok", msg=accepted)
        self.assertIn("'--comment=debug session'", accepted["command"])
        rejected = server_hpc.salloc_submit(partition="cpu", comment="a\nb", confirm=True)
        self.assertEqual(rejected.get("status"), "error")

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


class SlurmCancelTests(unittest.TestCase):
    """One job of the user's, approved as such: never a sweep, never someone else's."""

    def setUp(self) -> None:
        self._orig = (server_hpc._run_argv, server_hpc._current_user)
        self.argvs: list[list[str]] = []
        self.queue = "alice|RUNNING"
        server_hpc._current_user = lambda: "alice"

        def fake_argv(argv, timeout):
            self.argvs.append(argv)
            if argv[0] == "squeue":
                return {"status": "ok", "stdout": self.queue, "stderr": "", "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}
        server_hpc._run_argv = fake_argv

    def tearDown(self) -> None:
        server_hpc._run_argv, server_hpc._current_user = self._orig

    def _cancels(self) -> list[list[str]]:
        return [a for a in self.argvs if a[0] == "scancel"]

    def test_an_own_running_job_is_cancelled_by_id_alone(self) -> None:
        res = server_hpc.slurm_cancel(job_id="1234", confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["was"], "RUNNING")
        self.assertEqual(self._cancels(), [["scancel", "1234"]])

    def test_an_array_task_is_one_job(self) -> None:
        server_hpc.slurm_cancel(job_id="1234_5", confirm=True)
        self.assertEqual(self._cancels(), [["scancel", "1234_5"]])

    def test_anything_wider_than_one_job_is_refused_before_slurm(self) -> None:
        for job_id in ("", "-u alice", "1234 1235", "1234,1235", "--partition=gpu", "abc"):
            with self.subTest(job_id=job_id):
                res = server_hpc.slurm_cancel(job_id=job_id, confirm=True)
                self.assertEqual(res["status"], "error")
        self.assertEqual(self.argvs, [])

    def test_nothing_happens_without_confirm(self) -> None:
        res = server_hpc.slurm_cancel(job_id="1234")
        self.assertEqual(res["status"], "error")
        self.assertEqual(self.argvs, [])

    def test_someone_elses_job_is_refused(self) -> None:
        self.queue = "bob|PENDING"
        res = server_hpc.slurm_cancel(job_id="1234", confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("bob", res["error"])
        self.assertEqual(self._cancels(), [])

    def test_a_job_out_of_the_queue_says_how_it_ended(self) -> None:
        self.queue = ""
        orig = server_hpc._run_bash
        server_hpc._run_bash = _canned({"squeue": ("ok", ""), "sacct": ("ok", "COMPLETED")})
        self.addCleanup(setattr, server_hpc, "_run_bash", orig)
        res = server_hpc.slurm_cancel(job_id="1234", confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("COMPLETED", res.get("hint", ""))
        self.assertEqual(self._cancels(), [])

    def test_it_is_approved_and_not_held_like_a_submission(self) -> None:
        # Irreversible, so the client raises a card; not CLUSTER_SUBMIT, so the
        # local-validation hold on submissions does not apply to stopping a job.
        import asyncio
        from mimir.client.context.capabilities import (
            CLUSTER_SUBMIT, PLAN_BLOCKED, SENSITIVE, infer_tool_caps,
        )
        tools = asyncio.run(server_hpc.mcp.list_tools())
        caps = infer_tool_caps(next(t for t in tools if t.name == "slurm_cancel")).capabilities
        self.assertIn(SENSITIVE, caps)
        self.assertIn(PLAN_BLOCKED, caps)
        self.assertNotIn(CLUSTER_SUBMIT, caps)


if __name__ == "__main__":
    unittest.main()
