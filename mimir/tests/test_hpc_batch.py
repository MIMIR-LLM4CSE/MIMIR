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
import shlex
import sys
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "hpc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_hpc  # noqa: E402
import slurm_digest  # noqa: E402
import slurm_site  # noqa: E402


def _only_local(case: unittest.TestCase) -> None:
    """Make this machine's own Slurm the one scheduler for the rest of *case*.

    The inventory is the local probe plus whatever a session has learned, so a test
    that does not pin it reads the developer's machine: with ``sbatch`` on the box the
    answer is one scheduler, without it none, and a submission would be refused for
    having nowhere to go. Pinned here so what the tests exercise is the resolution and
    not the host they happen to run on.
    """
    orig = slurm_site._HAS_SLURM
    slurm_site._HAS_SLURM = True
    case.addCleanup(lambda: setattr(slurm_site, "_HAS_SLURM", orig))


def _canned(mapping: dict):
    """Return a fake _run_bash whose output depends on the command substring.

    Records every call as ``fake.calls`` — ``(script, stdin, host)`` — because what a
    tool submits, how it submits it, and which scheduler it went to are three things
    worth asserting on, and only this sees them.
    """
    def fake(script: str, timeout: int, stdin: str | None = None, *,
             host: str = "") -> dict:
        fake.calls.append((script, stdin, host))
        for key, (status, out) in mapping.items():
            if key in script:
                return {"status": status, "stdout": out, "stderr": "",
                        "returncode": 0 if status == "ok" else 1}
        return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}
    fake.calls: list = []
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

    def test_a_job_the_queue_calls_finished_is_finished(self) -> None:
        """Following it further is following something that cannot change again.

        A watcher drops a run the moment its state is terminal, so recognising the
        state *is* the cleanup: until these were read as terminal, a cancelled job was
        polled for as long as the process lived.
        """
        for state, expected in (("COMPLETED", "done"),
                                ("FAILED", "crashed"),
                                ("CANCELLED", "crashed"),
                                ("CANCELLED BY 1234", "crashed"),
                                ("TIMEOUT", "crashed"),
                                ("OUT_OF_MEMORY", "crashed"),
                                ("NODE_FAIL", "crashed")):
            with self.subTest(state=state):
                self.assertEqual(self._state({"squeue": ("ok", state)}), expected)

    def test_a_site_with_no_accounting_still_settles_its_jobs(self) -> None:
        """The case this exists for.

        Not every cluster runs Slurm accounting, so there is no second answer to settle
        a job with — and a state the queue reports but this client did not classify used
        to fall through to "in the queue, so still running".
        """
        state = self._state({"squeue": ("ok", "CANCELLED"), "sacct": ("error", "")})
        self.assertEqual(state, "crashed")

    def test_the_queue_is_believed_without_asking_accounting_at_all(self) -> None:
        """One round trip rather than two, for every finished job."""
        fake = _canned({"squeue": ("ok", "COMPLETED")})
        server_hpc._run_bash = fake
        server_hpc._normalized_job_state("123")
        self.assertEqual([c[0].split()[0] for c in fake.calls], ["squeue"])

    def test_an_array_with_one_task_cancelled_is_not_an_array_that_completed(self) -> None:
        self.assertEqual(
            self._state({"squeue": ("ok", "COMPLETED\nCANCELLED\nCOMPLETED")}),
            "crashed")

    def test_a_finished_task_beside_a_running_one_keeps_the_job_live(self) -> None:
        self.assertEqual(
            self._state({"squeue": ("ok", "COMPLETED\nRUNNING")}), "running")

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
        # What else of the user's is running is asked through the shared seam, not
        # through this module's _run_bash — so it is stubbed separately or every case
        # here shells out to a real scheduler and waits out its timeout.
        self._orig_queued = server_hpc.slurm_queued_by_host
        server_hpc.slurm_queued_by_host = lambda *_a, **_k: {"": []}
        self.addCleanup(lambda: setattr(server_hpc, "slurm_queued_by_host",
                                        self._orig_queued))
        # One scheduler, this machine's own, so resolution has nothing to refuse and
        # the real slurm_site.resolve_host is what runs. Forced rather than mocked: the
        # probe is a fact about the developer's machine, which a test may not inherit.
        _only_local(self)
        self._orig_run = server_hpc._run_argv
        # Everything a submission runs goes through _run_bash now — the submit itself
        # (the script on sbatch's stdin) and the question of what else of the user's is
        # running unaccounted for. Stubbed for the whole class: unstubbed, every case
        # shells out to a real scheduler through a login shell and waits out its
        # timeout. The piped scripts are kept, since what was submitted is the thing
        # several cases are about.
        self._orig_bash = server_hpc._run_bash
        self.piped: list[str] = []

        def _fake_bash(script, timeout, stdin=None, *, host=""):
            self.piped.append(stdin or "")
            if "sbatch" in script:
                return {"status": "ok", "stdout": "Submitted batch job 4242",
                        "stderr": "", "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}

        server_hpc._run_bash = _fake_bash
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
        server_hpc._run_bash = self._orig_bash

    def _submit(self, **kwargs) -> dict:
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
        server_hpc._normalized_job_state = \
            lambda jid, *, host="": ("running", "RUNNING", ["gpu007"])
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-b"}):
            status = server_hpc.slurm_job_status(job_id="4242")
        self.assertEqual(status["job_dir"], submitted)
        self.assertEqual(status["submitted_by_another_session"], "session-a")

    def test_a_job_of_this_session_is_not_labelled_as_another_s(self) -> None:
        self.addCleanup(setattr, server_hpc, "_normalized_job_state",
                        server_hpc._normalized_job_state)
        server_hpc._normalized_job_state = \
            lambda jid, *, host="": ("running", "RUNNING", ["gpu007"])
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "session-a"}):
            submitted = self._submit()["job_dir"]
            status = server_hpc.slurm_job_status(job_id="4242")
        self.assertEqual(status["job_dir"], submitted)
        self.assertNotIn("submitted_by_another_session", status)

    def _status_as(self, state: str, job_id: str = "4242") -> dict:
        self.addCleanup(setattr, server_hpc, "_normalized_job_state",
                        server_hpc._normalized_job_state)
        server_hpc._normalized_job_state = \
            lambda jid, *, host="": (state, state.upper(),
                                     ["gpu007"] if state == "running" else [])
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
            server_hpc._normalized_job_state = \
                lambda jid, *, host="": ("unknown", "", [])
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
        server_hpc._run_argv = lambda argv, t, host='': {
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
        server_hpc._run_bash = lambda script, t, stdin=None, host='': {
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
        server_hpc._run_argv = lambda argv, t, host='': {
            "status": "ok", "stdout": "Submitted batch job 7", "stderr": "",
            "returncode": 0}
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu",
                                       comment="mimir ratchet iter 12", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        self.assertEqual(res["comment"], "mimir ratchet iter 12")
        with open(res["batch_script"]) as fh:
            self.assertIn("#SBATCH --comment='mimir ratchet iter 12'", fh.read())

    def test_without_a_comment_the_directive_is_absent(self) -> None:
        server_hpc._run_argv = lambda argv, t, host='': {
            "status": "ok", "stdout": "Submitted batch job 8", "stderr": "",
            "returncode": 0}
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu", confirm=True)
        with open(res["batch_script"]) as fh:
            self.assertNotIn("--comment", fh.read())

    def test_a_newline_in_the_comment_is_refused_before_slurm(self) -> None:
        """It would otherwise forge a #SBATCH directive line of its own."""
        called: list = []
        server_hpc._run_argv = lambda argv, t, host='': called.append(argv) or {
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
        server_hpc._run_argv = lambda argv, t, host='': {
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
        server_hpc._run_argv = lambda argv, t, host='': {
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
        server_hpc._run_argv = lambda argv, t, host='': {"status": "error", "stderr": "denied"}
        rows = "cpu-n01|cpu|idle|64|773500|760000|(null)|2|32|1\ncpu-n01|cpu_night|idle|64|773500|760000|(null)|2|32|1\n"
        orig_bash = server_hpc._run_bash
        server_hpc._run_bash = lambda s, t, host='': {"status": "ok", "stdout": rows, "stderr": "", "returncode": 0}
        try:
            res = server_hpc.slurm_nodes()
            self.assertEqual(res["nodes_total"], 1)   # one node in two partitions, not two nodes
            self.assertEqual(res["architectures"], [])
            self.assertIn("architecture", res["degraded"])
        finally:
            server_hpc._run_bash = orig_bash


class SbatchTargetingTests(unittest.TestCase):
    def setUp(self) -> None:
        # A submission runs everything through _run_bash — the script goes to sbatch's
        # stdin — so that is the seam to stub: left alone, each case shells out to a
        # real scheduler. The piped script is what these cases are about.
        self._orig_queued = server_hpc.slurm_queued_by_host
        server_hpc.slurm_queued_by_host = lambda *_a, **_k: {"": []}
        self.addCleanup(lambda: setattr(server_hpc, "slurm_queued_by_host",
                                        self._orig_queued))
        _only_local(self)
        self._orig_bash = server_hpc._run_bash
        self.piped: list[str] = []

        def _fake_bash(script, timeout, stdin=None, *, host=""):
            self.piped.append(stdin or "")
            if "sbatch" in script:
                return {"status": "ok", "stdout": "Submitted batch job 7",
                        "stderr": "", "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}

        server_hpc._run_bash = _fake_bash
        self.addCleanup(lambda: setattr(server_hpc, "_run_bash", self._orig_bash))
        self._orig_dir = os.environ.get("MIMIR_STATE_DIR")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["MIMIR_STATE_DIR"] = self._tmp.name

    def tearDown(self) -> None:
        if self._orig_dir is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_dir
        self._tmp.cleanup()

    def test_constraint_nodelist_exclusive_reach_the_script(self) -> None:
        res = server_hpc.sbatch_submit(command="./bench", partition="cpu", constraint="icelake",
                                       nodelist="n[01-02]", exclusive=True, ntasks=4,
                                       confirm=True)
        self.assertEqual(res.get("status"), "ok", res.get("error"))
        script = self.piped[0]
        for want in ("--constraint=icelake", "--nodelist=n[01-02]", "--exclusive", "--ntasks=4"):
            self.assertIn(want, script)
        self.assertNotIn("--nodes=", script)   # 0 = scheduler default
        # The log goes where the job runs, so the two directives name one directory.
        self.assertIn("--chdir=", script)

    def test_bad_constraint_rejected(self) -> None:
        res = server_hpc.sbatch_submit(command="x", partition="cpu", constraint="a b",
                                       confirm=True)
        self.assertEqual(res.get("status"), "error")


def _fake_scontrol(stdout: str):
    """Stands in for ``_run_argv``, recording which scheduler each call went to.

    The host is a keyword on the real one and the calls that matter here pass it, so a
    fake that dropped it would let a tool asking the wrong cluster pass. Readable
    afterwards as ``fake.hosts``.
    """
    def fake(argv, t, *, host=""):
        fake.hosts.append(host)
        if argv[:2] == ["scontrol", "show"]:
            return {"status": "ok", "stdout": stdout, "stderr": "", "returncode": 0}
        if argv[:1] == ["sbatch"]:
            return {"status": "ok", "stdout": "Submitted batch job 900", "stderr": "",
                    "returncode": 0}
        return {"status": "error", "stdout": "", "stderr": "", "returncode": 1}
    fake.hosts: list[str] = []
    return fake


_NODE_LSCPU = ("Architecture: x86_64\nModel name: Intel(R) Xeon(R) Platinum 8358\n"
               "Socket(s): 2\nCore(s) per socket: 32\nThread(s) per core: 1\n"
               "L3 cache: 96 MiB (2 instances)\nFlags: avx2 fma avx512f avx512bw avx512vl\n")


class NodeProfileTests(unittest.TestCase):
    """Probing a compute node on the node, and reading the result back."""

    def setUp(self) -> None:
        self._saved = {k: getattr(server_hpc, k) for k in (
            "_run_argv", "_run_bash", "_NODE_PROFILES_DIR",
            "_host_identity", "_local_profile_record")}
        self._tmp = tempfile.TemporaryDirectory()
        # Taken before anything here is set, which is what lets tearDown put the
        # environment back: captured after, it would restore this test's own job dir
        # and every later test would file its submissions into one shared directory.
        self._env = dict(os.environ)
        self._jobs_dir = os.path.join(self._tmp.name, "jobs")
        os.environ["MIMIR_HPC_JOBS_DIR"] = self._jobs_dir
        server_hpc._NODE_PROFILES_DIR = os.path.join(self._tmp.name, "profiles")
        server_hpc._run_argv = _fake_scontrol(_SCONTROL)
        server_hpc._run_bash = _canned({
            "sbatch": ("ok", "Submitted batch job 900"),
            "squeue": ("ok", "PENDING"),
        })
        # The host MIMIR runs on: a login node of another generation, on a newer OS.
        server_hpc._host_identity = lambda: {
            "arch": "x86_64", "os": {"id": "rhel", "version": "9.4", "glibc": "2.34"},
            "cpu": {"model": "Intel(R) Xeon(R) 6747P", "simd": {"avx2": True, "amx_tile": True}},
        }
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

    def test_a_profile_is_filed_under_the_scheduler_it_was_read_on(self) -> None:
        """Node names are per cluster, so a profile without one is a profile of nothing.

        Slurm's own node names are chosen per site: two schedulers each with a
        ``node01`` are two unrelated machines, and one flat directory had the second
        overwrite the first.
        """
        job_dir = self._submit()
        self._finish(job_dir, "cpu001", "full", **{"profile.json": '{"cpu": {}}'})
        server_hpc.slurm_node_profile(partition="cpu")
        filed = os.path.join(server_hpc._NODE_PROFILES_DIR, "local", "cpu001.json")
        self.assertTrue(os.path.exists(filed), filed)
        with open(filed) as fh:
            self.assertEqual(json.load(fh)["scheduler"], "local")

    def test_two_schedulers_with_the_same_node_name_do_not_collide(self) -> None:
        """The bug this layout exists to stop."""
        record = {"node": "node01", "probed_at": "x", "profile": {"cpu": {"model": "A"}}}
        server_hpc._write_node_profile("node01", record, "")
        server_hpc._write_node_profile(
            "node01", {**record, "profile": {"cpu": {"model": "B"}}}, "hpc1")
        self.assertEqual(
            server_hpc._read_node_profile("node01", "")["profile"]["cpu"]["model"], "A")
        self.assertEqual(
            server_hpc._read_node_profile("node01", "hpc1")["profile"]["cpu"]["model"],
            "B")

    def test_a_profile_at_the_root_is_read_for_no_scheduler(self) -> None:
        """It records no host, and reading it as the local one is the guess the
        per-scheduler layout exists to prevent. A node re-profiles on demand."""
        os.makedirs(server_hpc._NODE_PROFILES_DIR, exist_ok=True)
        with open(os.path.join(server_hpc._NODE_PROFILES_DIR, "node01.json"), "w") as fh:
            json.dump({"node": "node01", "profile": {"cpu": {"model": "old"}}}, fh)
        self.assertIsNone(server_hpc._read_node_profile("node01", ""))
        self.assertIsNone(server_hpc._read_node_profile("node01", "hpc1"))

    def test_a_remote_probe_is_submitted_on_stdin_and_writes_on_the_far_side(self) -> None:
        """A remote scheduler cannot read a path on this machine.

        Neither the script nor the directory it writes into can be local: the first
        goes over on stdin, the second is a path the script makes on the node, under
        the state home the rest of the toolkit already uses there.
        """
        server_hpc._run_bash.calls.clear()
        res = server_hpc.slurm_probe_node(partition="cpu", scheduler="hpc1",
                                          confirm=True)
        self.assertEqual(res.get("status"), "ok")
        script, stdin, host = next(c for c in server_hpc._run_bash.calls
                                   if "sbatch" in c[0])
        self.assertEqual(script, "sbatch")
        self.assertEqual(host, "hpc1")
        self.assertIn("#SBATCH", stdin or "")
        self.assertIn("mkdir -p", stdin)
        self.assertIn("MIMIR_STATE_HOME", stdin)
        # And the marker says where to go looking for the result.
        with open(os.path.join(res["job_dir"], "probe.json")) as fh:
            marker = json.load(fh)
        self.assertEqual(marker["host"], "hpc1")
        self.assertIn("node_probes", marker["work_dir"])

    def test_a_local_probe_still_writes_in_its_own_submission_directory(self) -> None:
        res = server_hpc.slurm_probe_node(partition="cpu", confirm=True)
        with open(os.path.join(res["job_dir"], "probe.json")) as fh:
            marker = json.load(fh)
        self.assertEqual(marker["host"], "")
        self.assertEqual(marker["work_dir"], res["job_dir"])

    def test_a_remote_result_is_read_back_over_one_round_trip(self) -> None:
        """Four files, one connection: a call per file would be four ssh handshakes
        per probe, per panel refresh."""
        res = server_hpc.slurm_probe_node(partition="cpu", scheduler="hpc1",
                                          confirm=True)
        far_side = {"mode": "partial", "node": "cpu001",
                    "fallback.txt": "==uname\nx86_64\n==nproc\n64\n"}

        def fake(script, timeout, stdin=None, *, host=""):
            if "sbatch" in script:
                return {"status": "ok", "stdout": "Submitted batch job 900",
                        "stderr": "", "returncode": 0}
            if "==mode" in script:
                fake.reads.append(host)
                body = "".join(f"=={n}\n{far_side.get(n, '')}\n"
                               for n in server_hpc._PROBE_FILES)
                return {"status": "ok", "stdout": body, "stderr": "", "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}
        fake.reads = []
        server_hpc._run_bash = fake

        server_hpc.slurm_node_profile(partition="cpu", scheduler="hpc1")
        self.assertEqual(fake.reads, ["hpc1"], "one read, on the right scheduler")
        filed = os.path.join(server_hpc._NODE_PROFILES_DIR, "hpc1", "cpu001.json")
        self.assertTrue(os.path.exists(filed), filed)
        with open(filed) as fh:
            record = json.load(fh)
        self.assertEqual(record["scheduler"], "hpc1")
        self.assertTrue(record["partial"])
        self.assertIn("over ssh", record["partial_reason"])
        self.assertEqual(record["profile"]["cpu"]["logical_cpus"], 64)
        self.assertEqual(os.path.exists(
            os.path.join(server_hpc._NODE_PROFILES_DIR, "local", "cpu001.json")), False)

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
        self.assertFalse(os.listdir(self._jobs_dir) if os.path.isdir(
            self._jobs_dir) else [])
class SlurmCancelTests(unittest.TestCase):
    """One job of this conversation's, approved as such: never a sweep, never another's."""

    def setUp(self) -> None:
        self._orig = (server_hpc._run_argv, server_hpc._current_user,
                      server_hpc._run_bash)
        self.argvs: list[list[str]] = []
        self.queue = "alice|RUNNING"
        server_hpc._current_user = lambda: "alice"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, {"MIMIR_STATE_DIR": tmp.name,
                                      "MIMIR_SESSION_ID": "s-a"}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MIMIR_HPC_JOBS_DIR", None)
        self._submit("1234")
        # A cancellation finds the job before it sends anything, so this class needs
        # the scheduler to answer "running" as well: a scancel aimed at a scheduler
        # that never heard of the id either hits nothing or, the number existing on
        # two clusters, somebody else's job.
        _only_local(self)
        server_hpc._run_bash = _canned({"squeue": ("ok", "RUNNING")})

        def fake_argv(argv, timeout, *, host=""):
            self.argvs.append(argv)
            if argv[0] == "squeue":
                return {"status": "ok", "stdout": self.queue, "stderr": "", "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}
        server_hpc._run_argv = fake_argv

    def tearDown(self) -> None:
        (server_hpc._run_argv, server_hpc._current_user,
         server_hpc._run_bash) = self._orig

    def _submit(self, job_id: str, session: str = "s-a", host: str = "",
                **files: str) -> None:
        """File *job_id* as submitted by *session* on *host*, as a submission does."""
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": session}):
            slurm_site.record_job_dir(job_id, **{slurm_site.HOST_FILE: host, **files})

    def _cancels(self) -> list[list[str]]:
        return [a for a in self.argvs if a[0] == "scancel"]

    def _refused(self, job_id: str, **kwargs) -> dict:
        res = server_hpc.slurm_cancel(job_id=job_id, confirm=True, **kwargs)
        self.assertEqual(res["status"], "error")
        self.assertEqual(self._cancels(), [])
        return res

    def test_an_own_running_job_is_cancelled_by_id_alone(self) -> None:
        res = server_hpc.slurm_cancel(job_id="1234", confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["was"], "RUNNING")
        self.assertEqual(self._cancels(), [["scancel", "1234"]])

    def test_an_array_task_is_one_job(self) -> None:
        server_hpc.slurm_cancel(job_id="1234_5", confirm=True)
        self.assertEqual(self._cancels(), [["scancel", "1234_5"]])

    def test_another_sessions_job_of_the_same_user_is_refused(self) -> None:
        self._submit("2000", session="s-b")
        res = self._refused("2000")
        self.assertIn("not submitted by this conversation", res["error"])
        self.assertEqual(self.argvs, [])

    def test_a_job_no_session_here_submitted_is_refused(self) -> None:
        # Another workspace's, or a shell's: same Unix user, not this conversation's.
        self._refused("3000")
        self.assertEqual(self.argvs, [])

    def test_an_adopted_job_is_refused(self) -> None:
        self._submit("4000", adopted="\n")
        self._refused("4000")

    def test_a_job_recorded_on_another_scheduler_is_refused_here(self) -> None:
        self._submit("5000", host="cluster-b")
        self._refused("5000", scheduler="local")

    def test_a_job_without_a_recorded_scheduler_is_refused(self) -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s-a"}):
            slurm_site.record_job_dir("6000")
        self._refused("6000")

    def test_a_parent_cancels_its_sub_agents_job(self) -> None:
        self._submit("7000", session="s-a/subagents/sub-1/subagents/sub-2")
        res = server_hpc.slurm_cancel(job_id="7000", confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self._cancels(), [["scancel", "7000"]])

    def test_a_sub_agent_cannot_cancel_its_parents_or_siblings_job(self) -> None:
        self._submit("8000", session="s-a/subagents/sub-2")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s-a/subagents/sub-1"}):
            self._refused("1234")
            self._refused("8000")

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


class WhereAJobsOutputGoesTests(unittest.TestCase):
    """A job writes its output from the node it ran on, not from this host.

    The log used to be pointed at the submission's own directory under the state dir —
    the one directory in the arrangement that nothing guarantees the compute nodes can
    write. On a site whose home is not mounted on them, the job ran, succeeded, and its
    output went nowhere; an empty read then looked like a job that printed nothing.

    The rule that needs no configuration and no failure to learn: the output goes where
    the job *runs*. If the job could start at all, that directory is reachable from the
    node Slurm started it on — otherwise there would have been nothing to chdir into.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._state = os.path.join(self._tmp.name, "state")
        self._work = os.path.join(self._tmp.name, "scratch")
        os.makedirs(self._state)
        os.makedirs(self._work)
        env = patch.dict(os.environ, {"MIMIR_STATE_DIR": self._state}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MIMIR_SESSION_ID", None)
        os.environ.pop("MIMIR_HPC_JOBS_DIR", None)
        # What else of the user's is running is asked through the shared seam, not
        # through this module's _run_bash — so it is stubbed separately or every case
        # here shells out to a real scheduler and waits out its timeout.
        self._orig_queued = server_hpc.slurm_queued_by_host
        server_hpc.slurm_queued_by_host = lambda *_a, **_k: {"": []}
        self.addCleanup(lambda: setattr(server_hpc, "slurm_queued_by_host",
                                        self._orig_queued))
        # One scheduler, this machine's own, so resolution has nothing to refuse and
        # the real slurm_site.resolve_host is what runs. Forced rather than mocked: the
        # probe is a fact about the developer's machine, which a test may not inherit.
        _only_local(self)
        self._orig = server_hpc._run_bash
        self.piped: list[str] = []

        def _fake_bash(script, timeout, stdin=None, *, host=""):
            self.piped.append(stdin or "")
            if "sbatch" in script:
                return {"status": "ok", "stdout": "Submitted batch job 4242",
                        "stderr": "", "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}

        server_hpc._run_bash = _fake_bash

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig

    def _submit(self, **kwargs) -> dict:
        return server_hpc.sbatch_submit(command="echo hi", partition="cpu",
                                        confirm=True, **kwargs)

    def _script(self) -> str:
        return next(text for text in self.piped if "#SBATCH" in text)

    def test_the_job_runs_and_writes_in_the_same_directory(self) -> None:
        res = self._submit(work_dir=self._work)
        script = self._script()
        self.assertIn(f"#SBATCH --chdir={self._work}", script)
        self.assertIn(f"#SBATCH --output={os.path.join(self._work, 'slurm-%j.log')}",
                      script)
        self.assertEqual(res["log"], os.path.join(self._work, "slurm-4242.log"))
        self.assertEqual(res["work_dir"], self._work)

    def test_the_log_is_not_in_the_state_dir(self) -> None:
        # The whole failure: a path visible here and not from the compute node.
        res = self._submit(work_dir=self._work)
        self.assertNotIn(self._state, res["log"])
        self.assertNotIn(self._state, self._script())

    def test_it_defaults_to_where_this_server_runs(self) -> None:
        # Which is the workspace, and is where sbatch would have put the job anyway:
        # with no --chdir, Slurm starts a job in the directory it was submitted from.
        res = self._submit()
        self.assertEqual(res["work_dir"], os.path.realpath(os.getcwd()))

    def test_a_work_dir_that_is_not_there_is_refused(self) -> None:
        # The job starts in it, so a path that is not there is a submission that fails
        # at launch — after the queue wait.
        res = self._submit(work_dir=os.path.join(self._tmp.name, "nope"))
        self.assertEqual(res["status"], "error")

    def test_a_work_dir_it_cannot_write_is_refused_now(self) -> None:
        """Said before the queue wait, and not worked around.

        The tempting fallback is somewhere always writable, which on a cluster means
        ``/tmp`` — and ``/tmp`` is the compute node's own disk, so the output would
        land where this host can never read it. A shared directory that is not
        writable fails loudly; a node-local one fails silently, with an empty run that
        looks like a job which printed nothing.
        """
        locked = os.path.join(self._tmp.name, "readonly")
        os.makedirs(locked)
        os.chmod(locked, 0o500)
        self.addCleanup(os.chmod, locked, 0o700)
        res = self._submit(work_dir=locked)
        self.assertEqual(res["status"], "error")
        self.assertIn("not writable", res["error"])
        self.assertNotIn("/tmp", res["hint"])

    def test_the_script_is_piped_not_handed_over_by_path(self) -> None:
        """``sbatch`` reads its script from stdin when given no file.

        Which means nothing of ours has to exist on the host that runs it — the one
        thing a scheduler behind ``ssh`` cannot promise, and what otherwise has to be
        checked over ssh before a submission can be made at all.
        """
        res = self._submit(work_dir=self._work)
        self.assertIn("#!/bin/bash", self._script())
        self.assertIn("echo hi", self._script())
        # Still written down, for a reader asking what was actually submitted.
        with open(res["batch_script"]) as fh:
            self.assertEqual(fh.read(), self._script())

    def test_the_status_answer_names_the_log_the_submission_recorded(self) -> None:
        res = self._submit(work_dir=self._work)
        server_hpc._run_bash = _canned({"squeue": ("ok", "RUNNING")})
        status = server_hpc.slurm_job_status(job_id="4242")
        self.assertEqual(status["log"], res["log"])

    def test_a_finished_job_with_no_log_says_what_that_means(self) -> None:
        self._submit(work_dir=self._work)
        server_hpc._run_bash = _canned({"squeue": ("ok", ""),
                                        "sacct": ("ok", "COMPLETED")})
        status = server_hpc.slurm_job_status(job_id="4242")
        self.assertEqual(status["state"], "done")
        self.assertIn("work_dir", status["log_missing"])
        # Not a setting to go and change.
        self.assertNotIn("MIMIR_", status["log_missing"])


class HowACommandReachesItsSchedulerTests(unittest.TestCase):
    """Running a command where Slurm is, which is not always on this host.

    A login node that holds the editor's remote session while the scheduler lives
    behind ``ssh`` is an ordinary arrangement, and on one of those every cluster tool
    answered "sbatch is not available" — so the work was submitted by hand through the
    shell, where nothing writes a descriptor. Nothing was watched, no wake was ever
    owed, and the server stopped on its idle TTL with the user's job running.

    *Which* scheduler a call goes to is the caller's, resolved from what a tool was
    given against the session's inventory — see ``LearnedSchedulersTests`` in
    ``test_bash_background`` for how that inventory is built. These pin how a command
    is run once the host is known. The seam is shared (``_shared/slurm_site``) because
    all three servers ask the same question.
    """

    def test_local_is_run_through_a_login_shell(self) -> None:
        # The login shell is what puts the scheduler's own module files on PATH — the
        # difference between squeue and "command not found" on most sites.
        self.assertEqual(slurm_site.argv_for("squeue -h"),
                         ["bash", "-lc", "squeue -h"])

    def test_a_named_host_is_reached_over_ssh(self) -> None:
        argv = slurm_site.argv_for("squeue -u $USER", host="host-a")
        self.assertEqual(argv[0], "ssh")
        self.assertIn("host-a", argv)
        # A poller that can be asked for a password hangs for the timeout on every
        # tick, and an ssh that reads a server's stdin eats its MCP traffic.
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("-n", argv)
        # $USER has to reach the far side unexpanded: it is the remote account the
        # query is about.
        self.assertIn("squeue -u $USER", argv[-1])

    def test_a_piped_call_keeps_its_stdin(self) -> None:
        # ``-n`` closes stdin, which is right for every call but the one whose stdin
        # is the point: a submission hands sbatch its script that way.
        argv = slurm_site.argv_for("sbatch", host="host-a", piped=True)
        self.assertNotIn("-n", argv)
        self.assertIn("BatchMode=yes", argv)

    def test_no_ssh_options_of_our_own(self) -> None:
        # An identity file, a port, a jump host are ~/.ssh/config's business — a
        # configuration mechanism the user already has, and one we have no business
        # duplicating with a setting of our own.
        argv = slurm_site.argv_for("squeue", host="host-a")
        self.assertEqual([a for a in argv if a == "-o"], ["-o", "-o"])
        self.assertEqual(argv.count("host-a"), 1)

    def test_an_interactive_allocation_is_refused_over_ssh(self) -> None:
        """A silent no-op is worse than a plain no.

        ``salloc`` wants a terminal on the host that has the scheduler, and over ssh
        there is none: it would take the allocation, find no shell to run in it, and
        give it straight back — spending the request and reporting success at having
        done nothing.

        Asked of the scheduler the call *names*: the refusal is about where this one
        allocation would land, not about anything the session recorded earlier.
        """
        res = server_hpc.salloc_submit(partition="cpu", confirm=True,
                                       scheduler="host-a")
        self.assertEqual(res["status"], "error")
        self.assertIn("terminal", res["error"])
        self.assertIn("sbatch_submit", res["hint"])

    def test_an_interactive_allocation_is_allowed_where_slurm_is(self) -> None:
        # The other half: naming the local scheduler is not an ssh, so nothing is
        # refused — this one gets as far as actually running salloc.
        _only_local(self)
        ran: list[list[str]] = []
        orig = server_hpc._run_argv
        server_hpc._run_argv = lambda argv, t, host='': ran.append(argv) or {
            "status": "ok", "stdout": "granted", "stderr": "", "returncode": 0}
        self.addCleanup(lambda: setattr(server_hpc, "_run_argv", orig))
        res = server_hpc.salloc_submit(partition="cpu", confirm=True,
                                       scheduler="local")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(ran[0][0], "salloc")

    def test_an_argv_is_exec_d_as_it_stands_when_slurm_is_here(self) -> None:
        # The property the submitters are written for: they take their resources as
        # arguments and build the command themselves, so no shell is involved and no
        # argument can become a second command.
        self.assertEqual(slurm_site.exec_argv(["scancel", "1; rm -rf /"]),
                         ["scancel", "1; rm -rf /"])

    def test_an_argv_keeps_its_arguments_whole_over_ssh(self) -> None:
        """The same property in the form that survives ssh.

        ``ssh`` has no argv to hand over, only a command line for the far side to
        parse — so each element is quoted on the way out, and what the remote shell
        splits is exactly the argv built here.
        """
        argv = slurm_site.exec_argv(["scancel", "1; rm -rf /"], host="host-a")
        self.assertEqual(argv[0], "ssh")
        self.assertEqual(argv[-1], shlex.quote("scancel '1; rm -rf /'"))


class JobArraysTests(unittest.TestCase):
    """One id can be many jobs, and a site launcher submits exactly that.

    An array submits as ``15306621`` and runs as ``15306621_1 … _8``; squeue and sacct
    both answer with a line per element. Read as one line, an array whose first task
    has finished while seven still run reports as finished — waking its conversation
    on a run that has barely started.
    """

    def setUp(self) -> None:
        self._orig = server_hpc._run_bash

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig

    def _state(self, mapping: dict) -> tuple[str, str]:
        server_hpc._run_bash = _canned(mapping)
        state, raw, _nodes = server_hpc._normalized_job_state("15306621")
        return state, raw

    def test_one_task_still_running_keeps_the_array_live(self) -> None:
        state, _raw = self._state({"squeue": ("ok", "RUNNING\nRUNNING\nPENDING")})
        self.assertEqual(state, "running")

    def test_an_array_waiting_to_start_is_pending(self) -> None:
        state, _raw = self._state({"squeue": ("ok", "PENDING\nPENDING")})
        self.assertEqual(state, "pending")

    def test_a_finished_task_does_not_settle_the_whole_array(self) -> None:
        # The failure this exists for: the queue still holds live tasks, and sacct
        # already has a terminal line for the ones that are done.
        state, _raw = self._state({"squeue": ("ok", "RUNNING"),
                                   "sacct": ("ok", "COMPLETED\nCOMPLETED")})
        self.assertEqual(state, "running")

    def test_an_array_is_done_only_when_every_task_completed(self) -> None:
        state, _raw = self._state({"squeue": ("ok", ""),
                                   "sacct": ("ok", "COMPLETED\nCOMPLETED")})
        self.assertEqual(state, "done")

    def test_one_failed_task_makes_the_array_crashed(self) -> None:
        # An array where a task failed is not an array that completed, whichever line
        # happens to come first.
        state, raw = self._state({"squeue": ("ok", ""),
                                  "sacct": ("ok", "COMPLETED\nFAILED\nCOMPLETED")})
        self.assertEqual(state, "crashed")
        self.assertIn("FAILED", raw)


class ManySchedulersOneSessionTests(unittest.TestCase):
    """A job is not something a scheduler owns — it is something one of them is running.

    A machine reaches its own scheduler and whatever ``ssh`` gets to, and one
    conversation uses several: a reference run local, a sweep on another cluster. So
    the choice of where to submit is per call and refused rather than defaulted when
    it is a real choice; and finding a job again never depends on having been told
    where it is, because the watcher that polls it is re-armed from disk with nothing
    but an id.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._jobs = os.path.join(self._tmp.name, "hpc_jobs")
        env = patch.dict(os.environ, {"MIMIR_HPC_JOBS_DIR": self._jobs,
                                      "MIMIR_STATE_DIR": self._tmp.name}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        self._orig_bash = server_hpc._run_bash
        self.addCleanup(lambda: setattr(server_hpc, "_run_bash", self._orig_bash))
        self._orig_queued = server_hpc.slurm_queued_by_host
        server_hpc.slurm_queued_by_host = lambda *_a, **_k: {}
        self.addCleanup(lambda: setattr(server_hpc, "slurm_queued_by_host",
                                        self._orig_queued))
        # The inventory describes each host's hardware, so no real scheduler is asked.
        self._hardware: dict[str, str | None] = {}
        orig_capture = slurm_digest._capture
        slurm_digest._capture = self._capture
        self.addCleanup(setattr, slurm_digest, "_capture", orig_capture)
        slurm_digest.forget_cluster_digest()
        self.addCleanup(slurm_digest.forget_cluster_digest)

    def _capture(self, script, host, timeout):
        answer = self._hardware.get(host, _SCONTROL)
        if answer is None or not script.startswith("scontrol"):
            return None
        return answer

    def _reaching(self, *hosts: str) -> None:
        """Make the session reach exactly *hosts* ("" being this machine's own)."""
        orig = slurm_site._HAS_SLURM
        slurm_site._HAS_SLURM = "" in hosts
        self.addCleanup(lambda: setattr(slurm_site, "_HAS_SLURM", orig))
        for host in hosts:
            if host:
                slurm_site.record_host(host)

    def _answers(self, knows: dict[str, str]) -> list[tuple[str, str]]:
        """Let only the named hosts know a job, and record who was asked.

        The asking order is the assertion that matters: the hint first, then the rest
        — a poll runs on a timer for the life of a run, so a sweep on every tick is
        not a detail.
        """
        asked: list[tuple[str, str]] = []

        def fake(script, timeout, stdin=None, *, host=""):
            asked.append((host, script.split()[0]))
            state = knows.get(host, "")
            if "squeue" in script:
                return {"status": "ok", "stdout": state, "stderr": "",
                        "returncode": 0}
            return {"status": "ok", "stdout": "", "stderr": "", "returncode": 0}

        server_hpc._run_bash = fake
        return asked

    def test_a_submission_is_refused_rather_than_sent_somewhere_by_default(self) -> None:
        """The failure this whole shape exists for, caught at the one call that spends.

        With two schedulers reachable, "which one" is a decision about the work — the
        GPUs it needs, the filesystem its inputs are on. Picking the first silently is
        how a production run's submissions went to a cluster that had never heard of
        it, reported as an infrastructure problem for hours.
        """
        self._reaching("", "host-b")
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu",
                                       confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("local", res["error"])
        self.assertIn("host-b", res["error"])
        self.assertIn("scheduler=", res["error"])

    def test_one_scheduler_needs_no_naming(self) -> None:
        # The ordinary site: nothing to choose between, so nothing to say.
        self._reaching("")
        self._answers({})
        res = server_hpc.sbatch_submit(command="echo hi", partition="cpu",
                                       confirm=True, work_dir=self._tmp.name)
        self.assertEqual(res["status"], "error")  # the fake sbatch prints no id
        self.assertNotIn("scheduler=", res.get("error", ""))

    def test_a_job_is_found_on_the_scheduler_that_knows_it(self) -> None:
        """An id alone is enough, which is what the wake needs.

        A watcher re-armed after a window reload polls slurm_job_status(job_id) with
        no host — it has none to pass. So the answer has to be found: the descriptor's
        record first, and failing that every reachable scheduler.
        """
        self._reaching("", "host-b")
        asked = self._answers({"host-b": "RUNNING"})
        res = server_hpc.slurm_job_status(job_id="7001")
        self.assertEqual(res["state"], "running")
        self.assertEqual(res["scheduler"], "host-b")
        # In inventory order, and each host asked out before the next is tried: a
        # scheduler that does not know the id answers squeue and then sacct.
        self.assertEqual(list(dict.fromkeys(host for host, _ in asked)),
                         ["", "host-b"])

    def test_what_the_sweep_found_is_written_down_for_the_next_poll(self) -> None:
        # Polls run on a timer for the life of a run, so the sweep must be paid once.
        self._reaching("", "host-b")
        adopt = self._answers({"host-b": "RUNNING"})
        watched = server_hpc.slurm_watch(job_id="7001")
        self.assertEqual(watched["scheduler"], "host-b")
        with open(os.path.join(watched["job_dir"], "slurm_host")) as fh:
            self.assertEqual(fh.read().strip(), "host-b")
        asked = self._answers({"host-b": "RUNNING"})
        server_hpc.slurm_job_status(job_id="7001")
        self.assertEqual({host for host, _ in asked}, {"host-b"},
                         "the recorded host is a cache, and a poll that has one does "
                         "not sweep")
        self.assertTrue(adopt)

    def test_the_watcher_handle_stays_poll_able_by_id_alone(self) -> None:
        """The contract the client depends on, and the reason it needs no change.

        The watcher treats the descriptor as opaque and re-arms from disk with
        ``{"job_id": id}``. A handle carrying a host would be a handle the re-arm
        cannot rebuild.
        """
        self._reaching("", "host-b")
        self._answers({"host-b": "RUNNING"})
        res = server_hpc.slurm_watch(job_id="7001")
        self.assertEqual(res["background_jobs"][0]["status_op"],
                         {"tool": "slurm_job_status", "args": {"job_id": "7001"}})

    def test_a_job_no_scheduler_knows_is_refused_naming_the_ones_asked(self) -> None:
        self._reaching("", "host-b")
        self._answers({})
        res = server_hpc.slurm_watch(job_id="7001")
        self.assertEqual(res["status"], "error")
        self.assertIn("host-b", res["error"])
        self.assertIn("local", res["error"])

    def test_the_inventory_lists_what_can_be_submitted_to(self) -> None:
        self._reaching("", "host-b")
        res = server_hpc.slurm_schedulers()
        self.assertEqual([entry["scheduler"] for entry in res["schedulers"]],
                         ["local", "host-b"])
        self.assertEqual([entry["local"] for entry in res["schedulers"]],
                         [True, False])
        self.assertIn("names its scheduler", res["note"])

    def test_each_entry_says_what_its_machines_are(self) -> None:
        """Half the choice is hardware, and it has to be here to be made here.

        A model that must spend a slurm_nodes call per candidate before it can pick
        one picks without looking.
        """
        self._reaching("", "host-b")
        entries = server_hpc.slurm_schedulers()["schedulers"]
        self.assertIn("node type", entries[0]["hardware"])
        self.assertIn("x86_64", entries[0]["hardware"])
        self.assertEqual(len(entries[1]["node_types"]), 2)

    def test_a_silent_host_is_listed_without_being_called_empty(self) -> None:
        """Unknown hardware is not absent hardware, and neither is an error."""
        self._hardware["host-b"] = None
        self._reaching("", "host-b")
        entries = server_hpc.slurm_schedulers()["schedulers"]
        self.assertIn("node type", entries[0]["hardware"])
        self.assertEqual(entries[1]["hardware"], "")
        self.assertIn("unknown, not absent", entries[1]["hardware_note"])

    def test_a_host_is_verified_before_it_is_added(self) -> None:
        # Nothing is recorded on a guess: an unreachable host in the inventory is a
        # destination that swallows submissions.
        self._reaching("")
        server_hpc._run_bash = lambda script, t, stdin=None, host="": {
            "status": "error", "stderr": "ssh: Could not resolve hostname",
            "returncode": 255}
        res = server_hpc.slurm_schedulers(add="unreachable-host")
        self.assertEqual(res["status"], "error")
        self.assertNotIn("unreachable-host", slurm_site.hosts())

    def test_a_host_that_answers_is_added_and_can_be_dropped(self) -> None:
        self._reaching("")
        self._answers({})
        added = server_hpc.slurm_schedulers(add="host-b")
        self.assertIn("host-b", slurm_site.hosts())
        self.assertIn("added", added["added"]["note"])
        dropped = server_hpc.slurm_schedulers(forget="host-b")
        self.assertNotIn("host-b", slurm_site.hosts())
        self.assertIn("dropped", dropped["dropped"]["note"])

    def test_the_local_scheduler_is_not_a_record_to_drop(self) -> None:
        self._reaching("")
        res = server_hpc.slurm_schedulers(forget="local")
        self.assertIn("not a record", res["dropped"]["note"])
        self.assertEqual(slurm_site.hosts(), [""])


class TheQueueSaysWhatIsWatchedTests(unittest.TestCase):
    """Whether this workspace is watching a job is not a thing the scheduler knows.

    A job is tracked by the descriptor its submission wrote; one submitted any other
    way wrote none. The consequence is invisible from both ends — nothing polls it, no
    wake is owed, and the process stops on its idle TTL with the run still going — so
    the queue says it per row, and hands the set to the client where a nudge reads it.
    Two facts and no guessing, which is what makes it fit to be acted on.
    """

    def setUp(self) -> None:
        self._orig = server_hpc._run_bash
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._jobs = os.path.join(self._tmp.name, "hpc_jobs")
        env = patch.dict(os.environ, {"MIMIR_HPC_JOBS_DIR": self._jobs,
                                      "MIMIR_STATE_DIR": self._tmp.name}, clear=False)
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig

    def _recorded(self, job_id: str, host: str | None = "") -> None:
        """A descriptor as a submission or an adoption leaves it.

        *host* None is the one written before the scheduler was recorded beside the
        id, which still has to vouch for its job: an old submission reported as
        unwatched work is a reminder that cannot be acted on.
        """
        job_dir = os.path.join(self._jobs, f"20261010T120000Z-{job_id[-4:]}")
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "slurm_job_id"), "w") as fh:
            fh.write(job_id)
        if host is not None:
            with open(os.path.join(job_dir, "slurm_host"), "w") as fh:
                fh.write(host)

    def _queue(self, *rows: str, scheduler: str = "") -> dict:
        server_hpc._run_bash = _canned({"squeue": ("ok", "\n".join(rows))})
        return server_hpc.slurm_queue(scheduler=scheduler)

    def test_a_job_with_no_descriptor_is_reported_unwatched(self) -> None:
        res = self._queue("15306620|me|RUNNING|0:23|1:00:00|1|node1|smoke,rom")
        self.assertEqual(res["jobs"][0]["watched"], False)
        self.assertEqual(res["unwatched"], ["15306620"])
        self.assertIn("slurm_watch", res["unwatched_hint"])

    def test_a_recorded_job_is_not_in_the_set(self) -> None:
        self._recorded("15306620")
        res = self._queue("15306620|me|RUNNING|0:23|1:00:00|1|node1|smoke,rom")
        self.assertEqual(res["jobs"][0]["watched"], True)
        self.assertNotIn("unwatched", res)

    def test_the_same_id_on_another_scheduler_is_a_different_job(self) -> None:
        """Two clusters number independently, so an id alone is not an identity.

        15306620 is a watched production run on one and somebody else's sweep on the
        other. Keyed by id alone, the descriptor here would vouch for the job there:
        the queue would report it watched, the nudge would stay silent, and nothing
        would be polling it or owe a wake when it ended.
        """
        self._recorded("15306620", host="")        # on this machine's own scheduler
        res = self._queue("15306620|me|RUNNING|0:23|1:00:00|1|node1|someone-elses-sweep",
                          scheduler="host-b")
        self.assertEqual(res["jobs"][0]["watched"], False)
        self.assertEqual(res["unwatched"], ["15306620"])
        self.assertEqual(res["scheduler"], "host-b")

    def test_a_descriptor_that_did_not_record_a_host_still_covers_its_job(self) -> None:
        # Written before the host was kept beside the id. It cannot say where its job
        # is, and must not therefore be read as watching nothing.
        self._recorded("15306620", host=None)
        res = self._queue("15306620|me|RUNNING|0:23|1:00:00|1|node1|rom",
                          scheduler="host-b")
        self.assertEqual(res["jobs"][0]["watched"], True)
        self.assertNotIn("unwatched", res)

    def test_an_array_is_covered_by_its_base_id(self) -> None:
        # The id on disk is rarely the id in a row: an array is adopted or submitted
        # under its base and the queue then lists its tasks. Read literally, every task
        # of a watched array reads as unwatched and the nudge fires on all eight.
        self._recorded("15306621")
        res = self._queue(
            "15306621_1|me|RUNNING|0:11|1:00:00|4|node2|smoke,rom,C",
            "15306621_[2-8]|me|PENDING|0:00|1:00:00|4|(Dependency)|smoke,rom,C")
        self.assertEqual([row["watched"] for row in res["jobs"]], [True, True])
        self.assertNotIn("unwatched", res)

    def test_the_watched_and_the_unwatched_are_told_apart(self) -> None:
        self._recorded("15306620")
        res = self._queue(
            "15306620|me|RUNNING|0:23|1:00:00|1|node1|server",
            "15306621_3|me|RUNNING|0:11|1:00:00|4|node2|client")
        # Named by array, not by task: adoption is by array, so the answer says what
        # there is to hand over rather than how many rows it occupies.
        self.assertEqual(res["unwatched"], ["15306621"])

    def test_an_unwatched_array_is_one_thing_to_hand_over(self) -> None:
        # Eight task rows are one descriptor to write. Listed one by one, the reminder
        # built on this asks for eight calls that all land on "already recorded".
        res = self._queue(
            *[f"15306621_{n}|me|RUNNING|0:11|1:00:00|4|node2|client" for n in range(1, 9)])
        self.assertEqual(res["unwatched"], ["15306621"])
        self.assertEqual([row["watched"] for row in res["jobs"]], [False] * 8,
                         "the per-row flag still answers for the row")


class SlurmWatchTests(unittest.TestCase):
    """Adopting a job this server did not submit — the one it could never see.

    Their real submissions go through a site launcher inside a shell call, which prints
    the ids and exits. Nothing of MIMIR's is involved, so nothing is tracked: the job
    runs all night, no wake is owed, and the process stops on its idle TTL.
    """

    def setUp(self) -> None:
        # What else of the user's is running is asked through the shared seam, not
        # through this module's _run_bash — so it is stubbed separately or every case
        # here shells out to a real scheduler and waits out its timeout.
        self._orig_queued = server_hpc.slurm_queued_by_host
        server_hpc.slurm_queued_by_host = lambda *_a, **_k: {"": []}
        self.addCleanup(lambda: setattr(server_hpc, "slurm_queued_by_host",
                                        self._orig_queued))
        # One scheduler, this machine's own, so resolution has nothing to refuse and
        # the real slurm_site.resolve_host is what runs. Forced rather than mocked: the
        # probe is a fact about the developer's machine, which a test may not inherit.
        _only_local(self)
        self._orig = server_hpc._run_bash
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._jobs = os.path.join(self._tmp.name, "hpc_jobs")
        env = patch.dict(os.environ, {"MIMIR_HPC_JOBS_DIR": self._jobs}, clear=False)
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self) -> None:
        server_hpc._run_bash = self._orig

    def _watch(self, mapping: dict, job_id: str = "15306620", **kwargs) -> dict:
        server_hpc._run_bash = _canned(mapping)
        return server_hpc.slurm_watch(job_id=job_id, **kwargs)

    def test_a_running_job_is_recorded_and_watched(self) -> None:
        res = self._watch({"squeue": ("ok", "RUNNING")}, note="smoke ROM")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["state"], "running")
        # The descriptor the scan reads, which is the whole point of the tool.
        with open(os.path.join(res["job_dir"], "slurm_job_id")) as fh:
            self.assertEqual(fh.read().strip(), "15306620")
        # And the handle the client watcher arms on, in the shape sbatch_submit's has
        # but under the plural key: this call launched nothing, it handed a run over,
        # and the singular key is only ever read from a tool that declares a launch.
        self.assertEqual(res["background_jobs"][0]["job_key"], "15306620")
        self.assertEqual(res["background_jobs"][0]["status_op"]["tool"],
                         "slurm_job_status")
        self.assertNotIn("background_job", res)

    def test_an_array_base_id_is_accepted(self) -> None:
        res = self._watch({"squeue": ("ok", "RUNNING\nPENDING")}, job_id="15306621")
        self.assertEqual(res["state"], "running")

    def test_one_array_task_is_accepted(self) -> None:
        res = self._watch({"squeue": ("ok", "RUNNING")}, job_id="15306621_3")
        self.assertEqual(res["status"], "ok")

    def test_a_job_slurm_never_heard_of_is_refused(self) -> None:
        # A descriptor for a job that does not exist is worse than none: it holds the
        # process open until the staleness horizon for a run nobody is waiting on.
        res = self._watch({"squeue": ("ok", ""), "sacct": ("ok", "")})
        self.assertEqual(res["status"], "error")
        self.assertFalse(os.path.isdir(self._jobs), "it wrote a descriptor anyway")

    def test_a_job_name_is_not_a_job_id(self) -> None:
        res = self._watch({"squeue": ("ok", "RUNNING")}, job_id="smoke,rom,morTL")
        self.assertEqual(res["status"], "error")

    def test_a_finished_job_is_recorded_without_a_watcher(self) -> None:
        # A watcher put on a finished job reports it at its first poll, which is a
        # wake for news the caller is already holding.
        res = self._watch({"squeue": ("ok", ""), "sacct": ("ok", "COMPLETED")})
        self.assertEqual(res["state"], "done")
        self.assertNotIn("background_jobs", res)
        with open(os.path.join(res["job_dir"], "slurm_state")) as fh:
            self.assertEqual(fh.read().strip(), "done")

    def test_adopting_twice_does_not_make_a_second_descriptor(self) -> None:
        first = self._watch({"squeue": ("ok", "RUNNING")})
        again = self._watch({"squeue": ("ok", "RUNNING")})
        self.assertEqual(again["job_dir"], first["job_dir"])
        self.assertEqual(len(os.listdir(self._jobs)), 1)

    def _refuse_slurm(self):
        """A _run_bash that fails the test if anything asks a scheduler."""
        def fake(*_a, **_k):
            self.fail("it asked a scheduler for something the directory already knows")
        server_hpc._run_bash = fake

    def test_a_job_already_recorded_is_handed_over_again(self) -> None:
        """Recorded is not watched, and the answer used to say otherwise.

        A watcher is held by the agent that registered it and by nothing else, so a
        window reload leaves the directory saying "there is a job" with nobody polling
        it. Asking again is the moment that is repairable — and the answer carried no
        descriptor, so it repaired nothing while reporting the job as watched.
        """
        self._watch({"squeue": ("ok", "RUNNING")})
        self._refuse_slurm()          # the disk settles this, not two ssh round trips
        again = server_hpc.slurm_watch(job_id="15306620")
        self.assertEqual(again["status"], "ok")
        self.assertEqual(again["background_jobs"][0]["job_key"], "15306620")
        self.assertEqual(again["recorded_by"], "this conversation")

    def test_a_job_already_recorded_as_over_is_not_handed_over(self) -> None:
        # Its end was recorded by the poll that saw it, and a watcher put on it now
        # would report at its first poll — a wake for news already delivered.
        res = self._watch({"squeue": ("ok", ""), "sacct": ("ok", "COMPLETED")})
        self.assertEqual(res["state"], "done")
        self._refuse_slurm()
        again = server_hpc.slurm_watch(job_id="15306620")
        self.assertEqual(again["state"], "done")
        self.assertNotIn("background_jobs", again)

    def test_the_headline_promises_no_resume_of_its_own(self) -> None:
        # Whether anything watches the job is the client's to say, and it says so when
        # it registers a watcher. A server that guarantees a resume it does not perform
        # produces a model that ends its turn on a promise nobody is holding — which is
        # what every front-end with no watcher hook was handed here.
        res = self._watch({"squeue": ("ok", "PENDING")})
        self.assertNotIn("resumed", res["headline"])
        self.assertIn("15306620", res["headline"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
