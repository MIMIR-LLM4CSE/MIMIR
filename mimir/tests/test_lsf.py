"""LSF beside Slurm: the #BSUB header, bjobs/bhist read into run words, the registry's kind,
and the proxy's split eval submitted, read and cancelled through fake bsub/bjobs/bkill.
"""

import os
import tempfile
import unittest
from unittest import mock

from mimir.servers.proxy._lib import measure
from mimir.servers.workspace import bash_slurm
from mimir.shared import schedulers, cluster_site
from mimir.shared.schedulers import LSF, SLURM, bsub_header, lsf_minutes
from mimir.tests.support.proxy import _TmpStorageTest, _script, procs, server_proxy
from mimir.tests.support.scheduler import local_scheduler


def _lines(answers: dict[str, list[str]]):
    """A ``run_lines`` answering each command by its first word."""
    return lambda script: answers.get(script.split()[0], [])


class LsfStateTests(unittest.TestCase):
    def test_running_names_its_hosts_and_queue(self) -> None:
        found = LSF.state("12", _lines({"bjobs": ["RUN|8*c01:8*c02|normal"]}))
        self.assertEqual((found.state, found.nodes, found.partitions),
                         ("running", ["c01", "c02"], ["normal"]))

    def test_pending_and_suspended_words(self) -> None:
        self.assertEqual(LSF.state("1", _lines({"bjobs": ["PEND|-|normal"]})).state, "pending")
        self.assertEqual(LSF.state("1", _lines({"bjobs": ["PSUSP|-|normal"]})).state, "pending")
        self.assertEqual(LSF.state("1", _lines({"bjobs": ["USUSP|c01|normal"]})).state, "running")

    def test_ended_words(self) -> None:
        self.assertEqual(LSF.state("1", _lines({"bjobs": ["DONE|c01|normal"]})).state, "done")
        self.assertEqual(LSF.state("1", _lines({"bjobs": ["EXIT|c01|normal"]})).state, "crashed")

    def test_one_failed_element_makes_the_array_crashed(self) -> None:
        rows = ["DONE|c01|normal", "EXIT|c02|normal", "DONE|c03|normal"]
        self.assertEqual(LSF.state("7", _lines({"bjobs": rows})).state, "crashed")

    def test_a_cleaned_job_is_read_from_bhist_even_wrapped(self) -> None:
        """bhist wraps at 80 columns, mid-word too."""
        done = ["Job <5>, User <me>, Project <default>", "Mon Nov 16: Done succes",
                "                     sfully. The CPU time used is 2.0 seconds;"]
        self.assertEqual(LSF.state("5", _lines({"bhist": done})).state, "done")
        exited = ["Mon Nov 16: Exited with exit code 1. The CPU time used is 1.0 s"]
        self.assertEqual(LSF.state("5", _lines({"bhist": exited})).state, "crashed")

    def test_a_job_nobody_knows_is_unknown(self) -> None:
        self.assertEqual(LSF.state("5", _lines({})).state, "unknown")

    def test_submission_output_and_array_ids(self) -> None:
        self.assertEqual(LSF.submitted_id("Job <4242> is submitted to queue <normal>.\n"), "4242")
        self.assertIsNone(LSF.submitted_id("Request aborted by esub."))
        self.assertEqual(schedulers.array_base("4242[3]"), "4242")
        self.assertEqual(schedulers.array_base("4242_3"), "4242")
        self.assertEqual(bash_slurm.printed_ids("Job <9> is submitted to queue <q>.\n"
                                                "Submitted batch job 10\n"), ["9", "10"])


class BsubHeaderTests(unittest.TestCase):
    def _header(self, **kwargs) -> list[str]:
        base = dict(job_name="j", partition="normal", cpus_per_task=8,
                    wall_time="04:00:00", log_file="/w/out_%j.log")
        return bsub_header(**{**base, **kwargs})

    def test_the_placement_in_lsf_terms(self) -> None:
        lines = self._header(mem="32G", gpus=2, account="proj", nodes=1, ntasks=2,
                             constraint="a100&ib", nodelist="n[01-02]", exclusive=True,
                             comment='say "hi"', work_dir="/w")
        for want in ("#BSUB -q normal", "#BSUB -n 16", '#BSUB -R "span[hosts=1]"',
                     "#BSUB -W 4:00", "#BSUB -M 32GB", '#BSUB -gpu "num=2"', "#BSUB -P proj",
                     '#BSUB -R "select[a100 && ib]"', '#BSUB -m "n01 n02"', "#BSUB -x",
                     "#BSUB -Jd \"say 'hi'\"", "#BSUB -cwd /w", "#BSUB -oo /w/out_%J.log"):
            self.assertIn(want, lines)

    def test_defaults_add_nothing(self) -> None:
        text = "\n".join(self._header())
        for flag in ("-M", "-gpu", "-P", "select", "-m ", "-x", "-Jd", "-w ", "span"):
            self.assertNotIn(flag, text)

    def test_a_chained_job_waits_and_dies_with_its_build(self) -> None:
        lines = LSF.header(job_name="j", partition="q", cpus_per_task=1, wall_time="10",
                           log_file="/l", after_ok="77", kill_on_invalid_dep=True)
        self.assertIn('#BSUB -w "done(77)"', lines)
        self.assertIn("#BSUB -ti", lines)
        self.assertIn("#SBATCH --dependency=afterok:77",
                      SLURM.header(job_name="j", partition="q", cpus_per_task=1,
                                   wall_time="10", log_file="/l", after_ok="77"))

    def test_slurm_times_in_minutes_rounded_up(self) -> None:
        for given, minutes in (("04:00:00", 240), ("1-00:00:00", 1440), ("00:00:30", 1),
                               ("90", 90), ("1:30", 2), ("2-12", 3600)):
            self.assertEqual(lsf_minutes(given), minutes, given)

    def test_a_line_break_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._header(partition="normal\n#BSUB -q everything")


class LsfRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        local_scheduler(self, "")

    def test_a_host_keeps_its_kind_and_an_old_line_reads_as_slurm(self) -> None:
        with open(os.path.join(cluster_site.state_dir(), "slurm_hosts"), "w") as fh:
            fh.write("old-cluster\n")
        cluster_site.record_host("lsf-login", "lsf")
        self.assertEqual(cluster_site.hosts(), ["old-cluster", "lsf-login"])
        self.assertEqual(cluster_site.kind_of("old-cluster"), "slurm")
        self.assertIs(cluster_site.scheduler_for("lsf-login"), LSF)
        self.assertTrue(cluster_site.forget_host("lsf-login"))
        self.assertEqual(cluster_site.hosts(), ["old-cluster"])

    def test_an_ssh_call_says_which_scheduler_answered(self) -> None:
        self.assertEqual(cluster_site.scheduler_from_ssh(["ssh", "h", "bjobs -u me"]), ("h", "lsf"))
        self.assertEqual(cluster_site.scheduler_from_ssh(["ssh", "h", "squeue"]), ("h", "slurm"))
        self.assertEqual(cluster_site.scheduler_from_ssh(["ssh", "h", "ls"]), ("", ""))

    def test_the_local_scheduler_can_be_lsf(self) -> None:
        local_scheduler(self, "lsf")
        self.assertEqual(cluster_site.hosts(), [""])
        self.assertIs(cluster_site.scheduler_for(""), LSF)
        self.assertEqual(cluster_site.resolve_host("local"), ("", ""))


class LsfPeakMemoryTests(unittest.TestCase):
    def test_bjobs_max_mem_in_kilobytes(self) -> None:
        self.assertEqual(measure.lsf_mem_kb("1.5 Gbytes"), int(1.5 * 1024**2))
        self.assertEqual(measure.lsf_mem_kb("345 Mbytes"), 345 * 1024)
        self.assertIsNone(measure.lsf_mem_kb("-"))

    def test_an_lsf_job_asks_bjobs(self) -> None:
        with mock.patch.object(measure, "lsf_max_mem", return_value=(2048, None)) as asked:
            self.assertEqual(measure.scheduler_memory({"LSB_JOBID": "8"}), (2048, None))
        asked.assert_called_once_with("8")
        self.assertIsNone(measure.scheduler_memory({}))


class ProxyOnLsfTests(_TmpStorageTest):
    """An eval split into build and run, on a local LSF of fakes: builtins only, on a bare PATH."""

    def setUp(self) -> None:
        super().setUp()
        bin_dir = os.path.join(self.root, "bin")
        self.scripts = os.path.join(self.root, "scripts")
        os.makedirs(bin_dir)
        os.makedirs(self.scripts)
        counter = os.path.join(self.root, "counter")
        self.killed = os.path.join(self.root, "killed")
        _script(os.path.join(bin_dir, "bsub"),
                f"n=500\nif [ -f {counter} ]; then read n < {counter}; fi\nn=$((n+1))\n"
                f"echo $n > {counter}\n"
                f'while IFS= read -r line; do echo "$line"; done > {self.scripts}/$n\n'
                'echo "Job <$n> is submitted to queue <normal>."\n')
        _script(os.path.join(bin_dir, "bjobs"), 'echo "RUN|16*c07|normal"\n')
        _script(os.path.join(bin_dir, "bkill"), f'echo "$1" >> {self.killed}\n'
                                                'echo "Job <$1> is being terminated"\n')
        saved = os.environ["PATH"]
        self.addCleanup(os.environ.__setitem__, "PATH", saved)
        os.environ["PATH"] = bin_dir
        local_scheduler(self, "lsf")

    def _submitted(self) -> list[str]:
        return [open(os.path.join(self.scripts, n)).read()
                for n in sorted(os.listdir(self.scripts), key=int)]

    def test_a_split_eval_is_two_chained_lsf_jobs_read_and_cancelled_through_lsf(self) -> None:
        self._session(metadata={"build_cmd": "make all"})
        res = server_proxy.proxy_cluster(op="eval", partition="gpu", constraint="a100",
                                       build_partition="compile", confirm=True)
        self.assertEqual(res.get("status"), "ok", msg=res)
        self.assertEqual((res["build_job_id"], res["slurm_job_id"]), (501, 502))

        build_script, run_script = self._submitted()
        self.assertIn("#BSUB -q compile", build_script)
        self.assertNotIn("#SBATCH", build_script + run_script)
        self.assertIn("#BSUB -q gpu", run_script)
        self.assertIn('#BSUB -R "select[a100]"', run_script)
        self.assertIn('#BSUB -w "done(501)"', run_script)
        self.assertIn("#BSUB -ti", run_script)

        state = procs.run_state(res["run_dir"])
        self.assertEqual(state["state"], "running")

        cancelled = procs.cancel_run(res["run_dir"])
        self.assertEqual(cancelled.get("cancelled"), "lsf", msg=cancelled)
        with open(self.killed) as fh:
            self.assertEqual(fh.read().split(), ["501", "502"])


if __name__ == "__main__":
    unittest.main()
