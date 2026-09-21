"""Reading a machine the same way wherever the reading happens.

Covers the shared parsers (``_shared/cpu_facts``), the three execution contexts MIMIR
can find itself in (no scheduler, login node, inside an allocation), the shared batch
header (``_shared/slurm_script``) and the recognition of a shell ``srun`` that would
request a new allocation.

Run:
    python -m unittest mimir.tests.test_cpu_facts -v
"""

import sys
import unittest
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "servers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import cpu_facts  # noqa: E402
import slurm_script  # noqa: E402
from shell_paths import allocates_cluster  # noqa: E402


_LSCPU_X86 = """\
Architecture:                    x86_64
Model name:                      Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
Thread(s) per core:              1
Core(s) per socket:              32
Socket(s):                       2
L1d cache:                       3 MiB (64 instances)
L2 cache:                        80 MiB (64 instances)
L3 cache:                        96 MiB (2 instances)
NUMA node(s):                    2
Flags:                           fpu sse2 avx avx2 fma avx512f avx512bw avx512vl
"""

_LSCPU_ARM = """\
Architecture:                    aarch64
Model name:                      Neoverse-V2
Socket(s):                       1
Core(s) per socket:              72
Features:                        fp asimd sve sve2 bf16 i8mm
"""


class CpuFactsTests(unittest.TestCase):
    def test_x86_model_simd_and_caches(self) -> None:
        cpu = cpu_facts.cpu_facts("x86_64", _LSCPU_X86)
        self.assertIn("8358", cpu["model"])
        self.assertTrue(cpu["simd"]["avx512f"])
        self.assertFalse(cpu["simd"]["amx_tile"])
        # Cache sizes decide a blocking factor, and differ between login and compute.
        self.assertEqual(cpu["caches"]["l2"], "80 MiB (64 instances)")
        self.assertEqual(cpu["caches"]["l3"], "96 MiB (2 instances)")
        self.assertNotIn("l1i", cpu["caches"])

    def test_aarch64_reads_features_not_flags(self) -> None:
        cpu = cpu_facts.cpu_facts("aarch64", _LSCPU_ARM)
        self.assertEqual(cpu["simd"], {"asimd": True, "sve": True, "sve2": True,
                                       "bf16": True, "i8mm": True})
        self.assertNotIn("caches", cpu)

    def test_march_from_gcc_help_target(self) -> None:
        out = ("The following options are target specific:\n"
               "  -march=                     \ticelake-server\n"
               "  -mtune=                     \ticelake-server\n")
        self.assertEqual(cpu_facts.parse_march(out),
                         {"march": "icelake-server", "mtune": "icelake-server"})
        self.assertEqual(cpu_facts.parse_march("gcc: error"), {})

    def test_os_release_and_glibc(self) -> None:
        text = 'NAME="Red Hat"\nID="rhel"\nVERSION_ID="8.10"\nPRETTY_NAME="RHEL 8.10 (Ootpa)"\n'
        self.assertEqual(cpu_facts.parse_os_release(text),
                         {"id": "rhel", "version": "8.10", "name": "RHEL 8.10 (Ootpa)"})
        self.assertEqual(cpu_facts.parse_ldd_version("ldd (GNU libc) 2.28\nCopyright"), "2.28")

    def test_nvidia_csv_names_only(self) -> None:
        gpus = cpu_facts.parse_nvidia_csv("NVIDIA A100-SXM4-80GB\nNVIDIA A100-SXM4-80GB\n", ("name",))
        self.assertEqual([g["name"] for g in gpus], ["NVIDIA A100-SXM4-80GB"] * 2)

    def test_signature_separates_cpu_models_slurm_confuses(self) -> None:
        """Same core count and ISA, different CPU generation: Slurm's hardware key is
        the same, and the timing is not comparable. The signature must say so."""
        a = cpu_facts.cpu_facts("x86_64", _LSCPU_X86)
        b = dict(a, model="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz")
        self.assertNotEqual(cpu_facts.machine_signature("x86_64", a, []),
                            cpu_facts.machine_signature("x86_64", b, []))
        self.assertEqual(cpu_facts.machine_signature("x86_64", a, ["A100"]),
                         cpu_facts.machine_signature("x86_64", dict(a), ["A100"]))


class HostlistTests(unittest.TestCase):
    def test_ranges_lists_and_padding(self) -> None:
        self.assertEqual(cpu_facts.expand_hostlist("n[01-03,07],gpu1"),
                         ["n01", "n02", "n03", "n07", "gpu1"])
        self.assertEqual(cpu_facts.expand_hostlist("a[9-11]x"), ["a9x", "a10x", "a11x"])
        self.assertEqual(cpu_facts.expand_hostlist("single"), ["single"])


class ExecutionContextTests(unittest.TestCase):
    """The three places MIMIR can run from, each told apart by a deterministic signal."""

    def test_no_scheduler(self) -> None:
        self.assertEqual(cpu_facts.execution_context({}, "ws1", has_slurm=False),
                         {"context": "none"})

    def test_login_node(self) -> None:
        self.assertEqual(cpu_facts.execution_context({}, "login1", has_slurm=True)["context"],
                         "login")

    def test_inside_an_allocation_on_the_node(self) -> None:
        env = {"SLURM_JOB_ID": "77", "SLURM_JOB_NODELIST": "n[01-04]"}
        ctx = cpu_facts.execution_context(env, "n03.cluster", has_slurm=True)
        self.assertEqual(ctx["context"], "in_allocation")
        self.assertEqual(ctx["job_id"], "77")

    def test_login_node_holding_an_allocation_elsewhere(self) -> None:
        """salloc from a login node: a job id exists, but this host is not the node."""
        env = {"SLURM_JOB_ID": "77", "SLURM_JOB_NODELIST": "n[01-04]"}
        ctx = cpu_facts.execution_context(env, "login1", has_slurm=True)
        self.assertEqual(ctx["context"], "login")
        self.assertEqual(ctx["job_id"], "77")


class SbatchHeaderTests(unittest.TestCase):
    def test_targeting_lines(self) -> None:
        lines = slurm_script.sbatch_header(
            job_name="j", partition="cpu", cpus_per_task=4, wall_time="00:10:00",
            log_file="/tmp/x.log", constraint="icelake", nodelist="n[01-02]",
            exclusive=True, nodes=1, ntasks=8)
        for want in ("#SBATCH --constraint=icelake", "#SBATCH --nodelist=n[01-02]",
                     "#SBATCH --exclusive", "#SBATCH --nodes=1", "#SBATCH --ntasks=8"):
            self.assertIn(want, lines)

    def test_defaults_add_nothing(self) -> None:
        lines = slurm_script.sbatch_header(
            job_name="j", partition="cpu", cpus_per_task=1, wall_time="1:00:00",
            log_file="/tmp/x.log")
        self.assertFalse([ln for ln in lines if any(
            f in ln for f in ("--nodes", "--ntasks", "--constraint", "--nodelist",
                              "--exclusive", "--mem", "--gres", "--account"))])

    def test_validation_refuses_injection(self) -> None:
        self.assertIsNone(slurm_script.validate_target("skylake&ib", "n[01-04],g7"))
        self.assertIsNotNone(slurm_script.validate_target("a; rm -rf ~"))
        self.assertIsNotNone(slurm_script.validate_target(nodelist="n01 n02"))
        self.assertIsNotNone(slurm_script.validate_target(ntasks=0))

    def test_node_python_prefers_the_nodes_own_venv(self) -> None:
        script = "\n".join(slurm_script.node_python_lines())
        self.assertIn(".venv-", script)
        self.assertIn("$(uname -m)", script)
        self.assertIn(sys.executable, script)   # last resort, still tried
        self.assertEqual(slurm_script.node_python_lines(explicit="/opt/py"),
                         ["_MIMIR_PY=/opt/py"])


class ShellAllocationTests(unittest.TestCase):
    """`srun` outside an allocation is a submission; inside one it is a job step."""

    def test_srun_outside_an_allocation_allocates(self) -> None:
        for cmd in ("srun -p cpu make -j8", "make && srun -n 4 ./a.out",
                    "timeout 60 srun hostname", "OMP_NUM_THREADS=4 srun ./bench"):
            self.assertTrue(allocates_cluster(cmd, {}), cmd)

    def test_srun_inside_an_allocation_does_not(self) -> None:
        self.assertFalse(allocates_cluster("srun -n 4 ./a.out", {"SLURM_JOB_ID": "9"}))

    def test_srun_as_a_word_is_not_a_command(self) -> None:
        for cmd in ("echo srun", "cat srun.log", "grep srun notes.txt", "make -j"):
            self.assertFalse(allocates_cluster(cmd, {}), cmd)


class ProfileJsonTests(unittest.TestCase):
    """What a node probe runs: the host profile, printed, without serving MCP."""

    def test_prints_the_whole_profile_and_exits(self) -> None:
        import json
        import subprocess
        script = Path(__file__).resolve().parents[1] / "servers" / "hpc" / "server_platform.py"
        res = subprocess.run([sys.executable, str(script), "--profile-json"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stderr[-500:])
        profile = json.loads(res.stdout)
        for key in ("execution_context", "machine_signature", "os", "cpu", "march",
                    "memory", "gpu", "slurm", "modules", "toolchains", "conda_envs",
                    "virtualenvs"):
            self.assertIn(key, profile)
        self.assertIn(profile["execution_context"]["context"],
                      ("none", "login", "in_allocation"))


if __name__ == "__main__":
    unittest.main()
