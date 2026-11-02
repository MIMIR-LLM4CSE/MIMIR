"""A run's peak memory, measured by the server around each case: the largest resident set
of the case's process tree, which the code cannot forge, kept per case and as the run's
largest, gating requirements, and named when an accepted gain costs memory."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from mimir.servers.proxy import server_proxy
from mimir.servers.proxy._lib import command, layout, measure, proc_run, procs, profiling
from mimir.servers.proxy._ops import _eval_ratchet, eval_session
from mimir.tests.support.proxy import _TmpStorageTest, _eval


class SessionTests(_TmpStorageTest):
    def setUp(self) -> None:
        super().setUp()
        self.tracked = os.path.join(self.root, "tracked.py")
        self._megabytes(40)
        exe = os.path.join(self.root, "alloc.py")
        with open(exe, "w") as fh:
            fh.write(textwrap.dedent(f"""\
                import sys
                import numpy as np
                sys.path.insert(0, {self.root!r})
                from tracked import MB
                block = bytearray(MB * 2**20)
                block[::4096] = b'x' * len(block[::4096])
                import time; time.sleep(0.3)
                out = sys.argv[1] if sys.argv[1].endswith('.npz') else sys.argv[1] + '.npz'
                np.savez(out, field=np.ones(4))
                print('PROXY_METRICS_BEGIN')
                print('time_s=0.01')
                print('peak_rss_mb=1')
                print(f'output_file={{out}}')
                print('PROXY_METRICS_END')
                """))
        server_proxy.proxy_manage(op="register", name="alloc", executable_path=exe,
                                  run_cmd_template=f"{sys.executable} {{executable}} {{output_file}}",
                                  output_format="npz", confirm=True)
        server_proxy.proxy_manage(op="suite_define", name="allocb",
                                  cases=[{"case_id": "a", "proxy_name": "alloc"}], confirm=True)
        self.exe = exe
        # No profile: what is measured here is the run itself.
        patcher = mock.patch.object(profiling, "profile", return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _megabytes(self, mb: int) -> None:
        with open(self.tracked, "w") as fh:
            fh.write(f"MB = {mb}\n")

    def _init(self, requirements: list[dict]) -> None:
        res = _eval(op="init", proxy_name="alloc", benchmark_name="allocb", proxy_source_path=self.exe,
                    optimize_paths=[self.tracked], primary_metric="time_s", repeat=1,
                    requirements=requirements, confirm=True)
        self.assertEqual(res.get("status"), "ok", res)

    def _run(self) -> dict:
        cfg, error, run_dir, _notice = eval_session.prepare_run("alloc", "", False)
        self.assertIsNone(error)
        procs.update_opt_active_link("alloc", run_dir)
        from mimir.servers.proxy import _proxy_runner
        with mock.patch.object(sys, "argv", ["_proxy_runner", "--run-dir", run_dir]):
            _proxy_runner.main()
        return layout.read_json(os.path.join(run_dir, "metrics.json"))

    def test_the_peak_is_measured_and_a_reported_one_is_not_believed(self) -> None:
        self._init([{"metric": "time_s", "operator": "lt", "threshold": 100.0}])
        out = self._run()
        case = out["results"][0]["metrics"]
        self.assertGreater(case["peak_rss_mb"], 40)
        self.assertIn("peak_rss_mb", case["reserved_metrics_ignored"])
        self.assertEqual(out["peak_rss_mb"], case["peak_rss_mb"])
        self.assertEqual(case["processes"], 1)
        self.assertGreaterEqual(out["peak_rss_total_mb"], case["peak_rss_mb"])
        res = server_proxy.proxy_eval_status(op="results", proxy_name="alloc")
        self.assertEqual(res["final_metrics"]["peak_rss_mb"], out["peak_rss_mb"])

    def test_a_requirement_on_the_peak_gates_the_run(self) -> None:
        self._init([{"metric": "peak_rss_mb", "operator": "lt", "threshold": 20.0}])
        out = self._run()
        self.assertFalse(out["all_passed"])


class WarningTests(unittest.TestCase):
    def test_an_accepted_run_that_grew_its_memory_is_named(self) -> None:
        best = {"peak_rss_mb": 1000.0}
        warn = _eval_ratchet._memory_warning("accept", "time_s", best, {"peak_rss_mb": 1500.0})
        self.assertIn("+50%", warn)
        self.assertIn("level='memory'", warn)
        self.assertIsNone(_eval_ratchet._memory_warning("accept", "time_s", best, {"peak_rss_mb": 1100.0}))
        self.assertIsNone(_eval_ratchet._memory_warning("reject", "time_s", best, {"peak_rss_mb": 3000.0}))
        self.assertIsNone(_eval_ratchet._memory_warning("accept", "peak_rss_mb", best, {"peak_rss_mb": 3000.0}))
        self.assertIsNone(_eval_ratchet._memory_warning("accept", "time_s", {}, {"peak_rss_mb": 3000.0}))

    def test_an_mpi_run_is_judged_on_its_ranks_summed(self) -> None:
        """Each rank lighter, but twice as many: the sum is what grew."""
        best = {"peak_rss_mb": 1000.0, "peak_rss_total_mb": 4000.0}
        warn = _eval_ratchet._memory_warning("accept", "time_s", best,
                                             {"peak_rss_mb": 900.0, "peak_rss_total_mb": 7200.0})
        self.assertIn("peak_rss_total_mb grew +80%", warn)


_ALLOC = "import time; b = bytearray(60 * 2**20); b[::4096] = b'x' * len(b[::4096]); time.sleep(0.3)"


class SlurmTests(unittest.TestCase):
    """A job measures its solver as a local run does, and says so for the post-run step."""

    def test_the_launcher_records_exit_wall_and_peak(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            code = measure.main([run_dir, sys.executable, "-c", _ALLOC + "; raise SystemExit(3)"])
            self.assertEqual(code, 3)
            read = lambda name: procs.read_int_file(os.path.join(run_dir, name))
            self.assertEqual(read("returncode"), 3)
            self.assertIsNotNone(read("wall_time_ms"))
            self.assertGreater(read("peak_rss_kb"), 60 * 1024)

    def test_the_job_script_runs_the_solver_under_the_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            entry = {"name": "t", "executable_path": "/bin/true", "output_format": "npz",
                     "run_cmd_template": f"{sys.executable} -c \"{_ALLOC}\""}
            script = command.build_sbatch(entry, run_dir, "", partition="p", gpus=0, cpus_per_task=1,
                                          mem="1G", wall_time="00:10:00", account="", job_name="j",
                                          compare_to_reference="", python_exe=sys.executable)
            # The job's body, without Slurm's header or the post-run parse.
            body = "\n".join(line for line in script.splitlines()
                             if not line.startswith("#SBATCH") and "postrun" not in line)
            subprocess.run(["bash", "-c", body], check=True, capture_output=True)
            self.assertGreater(procs.read_int_file(os.path.join(run_dir, "peak_rss_kb")), 60 * 1024)
            self.assertEqual(procs.read_int_file(os.path.join(run_dir, "returncode")), 0)

    def test_without_a_mimir_python_the_shell_still_times_it(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            entry = {"name": "t", "executable_path": "/bin/true", "output_format": "npz",
                     "run_cmd_template": "true"}
            script = command.build_sbatch(entry, run_dir, "", partition="p", gpus=0, cpus_per_task=1,
                                          mem="1G", wall_time="00:10:00", account="", job_name="j",
                                          compare_to_reference="", python_exe="/nonexistent/python")
            body = "\n".join(line for line in script.splitlines()
                             if not line.startswith("#SBATCH") and "postrun" not in line)
            subprocess.run(["bash", "-c", body], check=True, capture_output=True)
            self.assertIsNotNone(procs.read_int_file(os.path.join(run_dir, "wall_time_ms")))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "peak_rss_kb")))



_RANK = "import time; b = bytearray(80 * 2**20); b[::4096] = b'x' * len(b[::4096]); time.sleep(0.5)"


class RanksTests(unittest.TestCase):
    """An MPI run's memory is its ranks' peaks summed, beside the largest one."""

    def test_the_ranks_of_a_local_mpirun_are_summed(self) -> None:
        if not shutil.which("mpirun"):
            self.skipTest("mpirun is not installed")
        out = proc_run.run(["mpirun", "--oversubscribe", "-np", "3", sys.executable, "-c", _RANK],
                           rusage=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out.returncode != 0:
            self.skipTest("mpirun cannot start ranks here")
        # The largest is one process; the total also holds the other ranks, each at its peak.
        self.assertGreater(out.peak_rss_kb, 80 * 1024)
        self.assertGreater(out.peak_rss_total_kb, out.peak_rss_kb + 2 * 80 * 1024)
        self.assertGreaterEqual(out.processes, 4)          # three ranks and mpirun

    def test_a_small_child_of_a_large_process_is_not_given_its_memory(self) -> None:
        """Linux hands the spawner's peak to the child's ru_maxrss across exec: a 10 MB
        program launched from a 300 MB process must still read about 10 MB."""
        ballast = bytearray(300 * 2**20)
        ballast[::4096] = b"x" * len(ballast[::4096])
        out = proc_run.run([sys.executable, "-c", "import time; time.sleep(0.3)"], rusage=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        del ballast
        self.assertLess(out.peak_rss_kb, 100 * 1024)

    def test_one_process_s_total_is_its_own_peak(self) -> None:
        out = proc_run.run([sys.executable, "-c", _RANK], rusage=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertEqual(out.processes, 1)
        self.assertLess(abs(out.peak_rss_total_kb - out.peak_rss_kb), 8 * 1024)

    def test_slurm_accounting_sums_every_step_s_tasks(self) -> None:
        text = ("123|||\n123.batch|1|2048K\n123.extern|2|0\n"
                "123.0|8|1048576K\n123.1|4|1.50G\n")
        self.assertEqual(measure.sacct_total(text), (8 * 1048576 + 4 * int(1.5 * 1024**2), 12))
        self.assertIsNone(measure.sacct_total("123|||\n123.batch|1|2048K\n"))

    def test_a_job_whose_ranks_slurm_counted_records_slurm_s_sum(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir, \
             mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), \
             mock.patch.object(measure, "slurm_steps", return_value=(9 * 1024**2, 16)):
            measure.main([run_dir, sys.executable, "-c", "pass"])
            read = lambda name: procs.read_int_file(os.path.join(run_dir, name))
            self.assertEqual((read("peak_rss_total_kb"), read("processes")), (9 * 1024**2, 16))


if __name__ == "__main__":
    unittest.main()
