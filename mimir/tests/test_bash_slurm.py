"""A shell command whose script reaches Slurm: found from its output or the queue, followed
through its jobs, told apart from a local run, and cancelled with them."""
import json
import os
import shutil
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mimir.shared import jobs, slurm_site
from mimir.shared.processes import session_argvs
from mimir.shared.slurm_site import SlurmJobState
from mimir.tests.support.scheduler import local_slurm
from mimir.servers.workspace import bash_exec, bash_jobs, bash_slurm, bash_validation, server_bash


class Cluster:
    """A scheduler reduced to what the code reads of it."""

    def __init__(self) -> None:
        self.states: dict[str, SlurmJobState] = {}
        self.submitted: dict[str, float] = {}
        self.silent = False
        self.scancelled: list[str] = []

    def job_state_on(self, job_id: str, host: str, timeout: float = 15.0):
        return None if self.silent else self.states.get(job_id, slurm_site.UNKNOWN_JOB)

    def queued_submissions(self, timeout: float = 15.0, *, host: str = ""):
        return dict(self.submitted)

    def read_lines(self, script: str, timeout: float = 15.0, *, host: str = ""):
        self.scancelled.append(script.split()[-1])
        return []


class ScriptJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="mimir_bash_slurm_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.work = os.path.realpath(os.path.join(self.root, "work"))
        os.makedirs(self.work)
        patches = [
            patch.dict(os.environ, {"MIMIR_STATE_DIR": os.path.join(self.root, "state")}),
            patch.object(server_bash, "WORKSPACE_ROOT", self.work),
            patch.object(bash_validation, "WORKSPACE_ROOT", self.work),
        ]
        for started in patches:
            started.start()
            self.addCleanup(started.stop)
        local_slurm(self)
        self.cluster = Cluster()
        for name in ("job_state_on", "queued_submissions", "read_lines"):
            stub = patch.object(slurm_site, name, getattr(self.cluster, name))
            stub.start()
            self.addCleanup(stub.stop)

    def _script(self, body: str, name: str = "submit.sh") -> str:
        path = os.path.join(self.work, name)
        Path(path).write_text("#!/bin/bash\n" + body + "\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return "./" + name

    def _launch(self, command: str) -> dict:
        started = bash_exec.launch_background(command, self.work, "")
        self.assertEqual(started["status"], "ok", started)
        return started

    def _status_once_exited(self, key: str) -> dict:
        deadline = time.time() + 10
        while time.time() < deadline:
            if os.path.exists(os.path.join(bash_jobs.job_dir(key), jobs.EXIT_CODE)):
                break
            time.sleep(0.02)
        return bash_jobs.state(key)

    def test_a_detached_script_reads_as_its_job_on_the_node(self):
        self.cluster.states["42"] = SlurmJobState("running", "RUNNING", ["n012"], ["gpu"])
        key = self._launch(self._script('echo "Submitted batch job 42"'))["job_key"]
        status = self._status_once_exited(key)
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["job_id"], "42")
        self.assertEqual(status["nodes"], ["n012"])
        self.assertEqual(status["partitions"], ["gpu"])
        self.assertEqual(status["scheduler"], "local")

    def test_the_run_ends_with_its_job_and_on_its_verdict(self):
        self.cluster.states["42"] = SlurmJobState("pending", "PENDING")
        key = self._launch(self._script('echo "Submitted batch job 42"'))["job_key"]
        self.assertEqual(self._status_once_exited(key)["state"], "pending")
        self.cluster.states["42"] = SlurmJobState("crashed", "FAILED")
        self.assertEqual(bash_jobs.state(key)["state"], "crashed")
        # Settled on disk: the scheduler is not asked again, and may have forgotten it.
        self.cluster.states.clear()
        self.assertEqual(bash_jobs.state(key)["state"], "crashed")

    def test_srun_waiting_in_the_queue_is_recognised(self):
        self.cluster.states["7"] = SlurmJobState("running", "RUNNING", ["n1"])
        key = self._launch(self._script(
            'echo "srun: job 7 queued and waiting for resources" >&2'))["job_key"]
        self.assertEqual(self._status_once_exited(key)["job_id"], "7")

    def test_a_silent_script_is_found_in_the_queue(self):
        self.cluster.states["9_1"] = SlurmJobState("running", "RUNNING", ["n3"])
        self.cluster.states["9"] = self.cluster.states["9_1"]
        script = self._script("sbatch --parsable job.sbatch >/dev/null")
        self.cluster.submitted["9_1"] = time.time()
        key = self._launch(script)["job_key"]
        self.assertEqual(self._status_once_exited(key)["slurm_jobs"], ["9"])

    def test_a_submission_outside_the_run_is_left_out(self):
        self.cluster.states["5"] = SlurmJobState("running", "RUNNING", ["n3"])
        self.cluster.submitted["5"] = time.time() - 3600
        key = self._launch(self._script("sbatch --parsable job.sbatch >/dev/null || true"))["job_key"]
        status = self._status_once_exited(key)
        self.assertEqual(status["state"], "done")
        self.assertNotIn("job_id", status)

    def test_a_script_that_names_no_slurm_never_asks_the_queue(self):
        with patch.object(slurm_site, "queued_submissions") as asked:
            key = self._launch(self._script("echo built"))["job_key"]
            status = self._status_once_exited(key)
        self.assertEqual(status["state"], "done")
        asked.assert_not_called()

    def test_a_local_script_keeps_its_pid(self):
        key = self._launch(self._script("sleep 30"))["job_key"]
        self.addCleanup(bash_jobs.stop, key)
        status = bash_jobs.state(key)
        self.assertEqual(status["state"], "running")
        self.assertIn("pid", status)
        self.assertNotIn("job_id", status)

    def test_an_id_read_back_from_an_old_log_is_not_a_job_of_this_run(self):
        self.cluster.states["3"] = SlurmJobState("done", "COMPLETED")
        Path(self.work, "old.out").write_text("Submitted batch job 3\n")
        key = self._launch("cat old.out")["job_key"]
        status = self._status_once_exited(key)
        self.assertEqual(status["state"], "done")
        self.assertNotIn("job_id", status)
        self.assertEqual(slurm_site.recorded_jobs(), set())

    def test_a_silent_scheduler_is_no_verdict(self):
        self.cluster.states["42"] = SlurmJobState("running", "RUNNING")
        key = self._launch(self._script('echo "Submitted batch job 42"'))["job_key"]
        self._status_once_exited(key)
        self.cluster.silent = True
        self.assertEqual(bash_jobs.state(key)["state"], "")
        self.cluster.silent = False
        self.assertEqual(bash_jobs.state(key)["state"], "running")

    def test_a_job_found_is_the_session_s_own(self):
        self.cluster.states["42"] = SlurmJobState("running", "RUNNING")
        key = self._launch(self._script('echo "Submitted batch job 42"'))["job_key"]
        self._status_once_exited(key)
        self.assertEqual(slurm_site.owned_job_hosts("42"), {""})
        self.assertEqual(slurm_site.unwatched(["42"], host=""), [])

    def test_stopping_the_run_cancels_its_jobs(self):
        self.cluster.states["42"] = SlurmJobState("running", "RUNNING")
        key = self._launch(self._script('echo "Submitted batch job 42"; sleep 30'))["job_key"]
        deadline = time.time() + 10
        while "job_id" not in bash_jobs.state(key) and time.time() < deadline:
            time.sleep(0.05)
        stopped = bash_jobs.stop(key)
        self.assertEqual(self.cluster.scancelled, ["42"])
        self.assertEqual(stopped["cancelled_slurm_jobs"], ["42"])

    def test_a_restart_sees_a_run_its_job_keeps_going(self):
        self.cluster.states["42"] = SlurmJobState("running", "RUNNING")
        key = self._launch(self._script('echo "Submitted batch job 42"'))["job_key"]
        self._status_once_exited(key)
        job_dir = bash_jobs.job_dir(key)
        self.assertTrue(jobs.follows_open_slurm_job(job_dir))
        filed = json.loads(Path(job_dir, jobs.SHELL_SLURM_JOBS).read_text())
        jobs.record_terminal_state(filed["jobs"]["42"]["dir"], "done")
        self.assertFalse(jobs.follows_open_slurm_job(job_dir))
        watcher = json.loads(Path(filed["jobs"]["42"]["dir"], jobs.WATCHER_FILE).read_text())
        self.assertEqual(watcher["job_key"], key)

    def test_a_blocking_script_hands_its_job_over(self):
        self.cluster.states["42"] = SlurmJobState("pending", "PENDING")
        result = server_bash.bash_run(self._script('echo "Submitted batch job 42"'))
        self.assertEqual(result["slurm_jobs"], ["42"])
        self.assertEqual([d["job_key"] for d in jobs.handed_over(result)], ["42"])
        self.assertEqual(slurm_site.owned_job_hosts("42"), {""})

    def test_a_scheduler_reached_through_ssh_is_learned(self):
        self.assertEqual(
            bash_slurm._schedulers("ssh -p 22 login1 ./run.sh", ["front1"]),
            ["", "login1", "front1"])

    def test_a_script_exec_ing_srun_over_ssh_is_found_through_its_process(self):
        # The host is a variable of the script and srun prints nothing: only the running
        # process says where the job went.
        local_slurm(self, False)
        self.cluster.states["777"] = SlurmJobState("running", "RUNNING", ["node017"], ["gpu"])
        stubs = os.path.join(self.root, "stubs")
        os.makedirs(stubs)
        Path(stubs, "ssh").write_text("#!/bin/bash\nexec -a ssh sleep 30\n")
        os.chmod(os.path.join(stubs, "ssh"), 0o755)
        script = self._script(
            f'export PATH={stubs}:$PATH\nFRONTEND="$1"\nPARTITION=gpu\nNODE=node017\n'
            'exec ssh "$FRONTEND" "/usr/bin/srun --partition=\'$PARTITION\' '
            '--nodelist=\'$NODE\' ./solver"', "launch.sh")
        self.cluster.submitted["777"] = time.time()
        key = self._launch(f"{script} front1")["job_key"]
        self.addCleanup(bash_jobs.stop, key)

        seen = [["bash", "launch.sh"],
                ["ssh", "front1", "/usr/bin/srun --partition='gpu' ./solver"]]
        with patch.object(bash_slurm, "session_argvs", lambda pid: seen):
            status = bash_jobs.state(key)
        self.assertEqual(status["job_id"], "777")
        self.assertEqual(status["scheduler"], "front1")
        self.assertEqual(status["nodes"], ["node017"])
        self.assertIn("front1", slurm_site.hosts())

    def test_the_processes_of_a_run_are_read_from_its_session(self):
        key = self._launch(self._script("sleep 30", "wait.sh"))["job_key"]
        self.addCleanup(bash_jobs.stop, key)
        deadline = time.time() + 5
        heads: list[str] = []
        while "sleep" not in heads and time.time() < deadline:
            heads = [argv[0] for argv in session_argvs(bash_jobs.state(key)["pid"])]
        self.assertIn("sleep", heads)
        self.assertEqual(session_argvs(999999999), [])

    def test_an_ssh_that_runs_no_slurm_reaches_no_scheduler(self):
        with patch.object(bash_slurm, "session_argvs",
                          lambda pid: [["ssh", "host", "make -j8"], ["make"]]):
            self.assertEqual(bash_slurm._running_slurm(1), [])

    def test_a_local_srun_is_seen_in_its_process(self):
        with patch.object(bash_slurm, "session_argvs",
                          lambda pid: [["/usr/bin/srun", "./a.out"]]):
            self.assertEqual(bash_slurm._running_slurm(1), [""])

    def test_a_script_is_read_where_the_chain_runs_it(self):
        os.makedirs(os.path.join(self.work, "sub"))
        self._script("srun ./a.out", "sub/go.sh")
        self.assertTrue(bash_slurm.runs_slurm_script("cd sub && ./go.sh", self.work))
        self.assertFalse(bash_slurm.runs_slurm_script("./go.sh", self.work))
        self.assertTrue(bash_slurm.runs_slurm_script("timeout 60 ./sub/go.sh", self.work))
        self.assertTrue(bash_slurm.runs_slurm_script("python3 -u sub/go.sh", self.work))


class StateOnTests(unittest.TestCase):
    def test_a_forgotten_job_is_not_a_silent_scheduler(self):
        answers = iter([None, ["42|COMPLETED|0:0"]])
        with patch.object(slurm_site, "read_lines", lambda *a, **k: next(answers)):
            found = slurm_site.job_state_on("42", "")
        self.assertIsNotNone(found)

    def test_nothing_answering_is_none(self):
        with patch.object(slurm_site, "read_lines", lambda *a, **k: None):
            self.assertIsNone(slurm_site.job_state_on("42", ""))


if __name__ == "__main__":
    unittest.main()
