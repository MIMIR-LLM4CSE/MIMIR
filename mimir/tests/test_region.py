"""The harness counts for nothing: the measured region it marks is timed and weighed by
the server in Python, C and Fortran alike, profiles set the harness's own work apart, and
the profilers are chosen by what the run runs, in front of rank 0 under MPI."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from mimir.servers.proxy import region as helpers
from mimir.servers.proxy._lib import execute, measure, metrics, proc_run, profiling
from mimir.servers.proxy._lib.profiling import Frame, callgrind, cpu, describe, frames
from mimir.servers.proxy._lib.profiling.target import split
from mimir.servers.proxy import server_proxy
from mimir.servers.proxy._ops import _eval_ratchet, eval_observe, eval_session
from mimir.servers.proxy._lib import procs
from mimir.shared.atomic import read_json
from mimir.tests.support.proxy import _TmpStorageTest, _eval

# 200 MB of inputs made outside the region, 50 MB allocated inside it.
_PY = """\
import sys, time
sys.path.insert(0, sys.argv[1])
from mimir_region import region
big = bytearray(200 * 2**20); big[::4096] = b"x" * len(big[::4096])
time.sleep(0.3)
with region():
    a = bytearray(50 * 2**20); a[::4096] = b"y" * len(a[::4096])
    time.sleep(0.2)
    del a
time.sleep(0.3)
"""
_C = """\
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mimir_region.h"
volatile char sink;
int main(void) {
    char *b = malloc(200 << 20); memset(b, 1, 200 << 20); sink = b[12345]; usleep(300000);
    mimir_region_begin();
    char *a = malloc(50 << 20); memset(a, 2, 50 << 20); sink = a[777]; usleep(200000); free(a);
    mimir_region_end();
    usleep(300000); free(b); return 0;
}
"""
_F90 = """\
program h
  use mimir_region
  real(8), allocatable :: b(:), a(:)
  allocate(b(200*131072)); b = 1
  call sleep(1)
  call mimir_region_begin()
  allocate(a(50*131072)); a = 2
  deallocate(a)
  call mimir_region_end()
  print *, b(5)
end program
"""


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def _write(self, name: str, text: str) -> str:
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def _measure(self, argv: list[str]):
        with open(os.devnull, "w") as out:
            return proc_run.run(argv, rusage=True, timeout=120, stdout=out, stderr=subprocess.STDOUT)


class RegionTests(_Tmp):
    """The server reads the clocks and the memory at each mark: what the harness does
    outside its region weighs nothing."""

    def _check(self, done, extra_mb: float = 50) -> None:
        r = done.region
        self.assertEqual(r["count"], 1)
        self.assertAlmostEqual(r["extra_kb"] / 1024, extra_mb, delta=8)
        # The run's own peak survives the mark's restart: the inputs and the region's.
        self.assertGreater(done.peak_rss_kb, 245 * 1024)
        self.assertGreater(r["peak_kb"], 245 * 1024)

    def test_a_python_harness(self) -> None:
        h = self._write("h.py", _PY)
        done = self._measure([sys.executable, h, helpers.HERE])
        self._check(done)
        self.assertGreater(done.region["time_s"], 0.18)
        self.assertLess(done.region["time_s"], 0.6)
        self.assertEqual(done.fingerprint["programs"], [os.path.basename(os.path.realpath(sys.executable))])

    def test_a_c_harness(self) -> None:
        if not shutil.which("gcc"):
            self.skipTest("gcc is not installed")
        exe = os.path.join(self.dir, "hc")
        subprocess.run(["gcc", "-O0", "-I", helpers.HERE, self._write("h.c", _C), "-o", exe], check=True)
        done = self._measure([exe])
        self._check(done)
        self.assertAlmostEqual(done.region["time_s"], 0.2, delta=0.1)

    def test_a_fortran_harness(self) -> None:
        if not shutil.which("gfortran"):
            self.skipTest("gfortran is not installed")
        exe = os.path.join(self.dir, "hf")
        subprocess.run(["gfortran", "-O0", "-J", self.dir, os.path.join(helpers.HERE, "mimir_region.f90"),
                        self._write("h.f90", _F90), "-o", exe], check=True)
        done = self._measure([exe])
        self._check(done)
        self.assertIn("gfortran", done.fingerprint["runtimes"])

    def test_outside_mimir_the_helpers_do_nothing(self) -> None:
        h = self._write("h.py", _PY)
        env = {k: v for k, v in os.environ.items() if k != "MIMIR_REGION"}
        subprocess.run([sys.executable, h, helpers.HERE], check=True, env=env, timeout=60)

    def test_a_run_that_marks_nothing_has_no_region(self) -> None:
        done = self._measure([sys.executable, "-c", "pass"])
        self.assertIsNone(done.region)

    def test_regions_in_turn_add_their_times(self) -> None:
        h = self._write("h.py", textwrap.dedent("""\
            import sys, time
            sys.path.insert(0, sys.argv[1])
            from mimir_region import region
            for _ in range(3):
                time.sleep(0.1)
                with region():
                    time.sleep(0.1)
            """))
        r = self._measure([sys.executable, h, helpers.HERE]).region
        self.assertEqual(r["count"], 3)
        self.assertAlmostEqual(r["time_s"], 0.3, delta=0.1)

    def test_a_batch_run_leaves_its_region_for_the_post_run_step(self) -> None:
        h = self._write("h.py", _PY)
        run_dir = os.path.join(self.dir, "run")
        os.makedirs(run_dir)
        with mock.patch.object(measure, "scheduler_memory", return_value=None):
            self.assertEqual(measure.main([run_dir, sys.executable, h, helpers.HERE]), 0)
        with open(os.path.join(run_dir, "stdout.log"), "w"):
            pass
        settled = execute._settle_metrics("", 1.0, 0, region=__import__("json").load(
            open(os.path.join(run_dir, proc_run.REGION_FILE))))
        self.assertEqual(settled["regions"], 1)
        self.assertAlmostEqual(settled["region_extra_mb"], 50, delta=8)
        self.assertTrue(os.path.isfile(os.path.join(run_dir, proc_run.PROCESSES_FILE)))

    def test_the_helpers_are_installed_beside_a_harness(self) -> None:
        placed = helpers.install(self.dir, ("python", "c", "fortran"))
        self.assertEqual(sorted(map(os.path.basename, placed)),
                         ["mimir_region.f90", "mimir_region.h", "mimir_region.py"])


class ObjectiveTests(unittest.TestCase):
    """time_s is the region's when there is one; nothing changes when there is none."""

    def test_the_region_s_time_is_time_s_and_a_report_is_set_aside(self) -> None:
        m = {"time_s": 0.001}
        metrics.normalize_time_metrics(m, 2.0, {"time_s": 0.4, "count": 2, "peak_kb": 2048,
                                                "peak_total_kb": 4096, "extra_kb": 1024})
        self.assertEqual((m["time_s"], m["region_time_s"], m["time_s_ignored"], m["regions"]), (0.4, 0.4, 0.001, 2))
        self.assertEqual((m["region_peak_rss_mb"], m["region_peak_rss_total_mb"], m["region_extra_mb"]), (2.0, 4.0, 1.0))

    def test_without_a_region_nothing_changes(self) -> None:
        m = {"time_s": 0.5}
        metrics.normalize_time_metrics(m, 2.0, None)
        self.assertEqual(m, {"time_s": 0.5, "wall_time_s": 2.0})

    def test_a_harness_cannot_forge_the_region_s_figures(self) -> None:
        m = execute._settle_metrics("PROXY_METRICS_BEGIN\nregion_time_s=0.0\nregion_extra_mb=0\nPROXY_METRICS_END\n",
                                    1.0, 0)
        self.assertEqual(m["reserved_metrics_ignored"], ["region_extra_mb", "region_time_s"])
        self.assertNotIn("region_time_s", m)

    def test_a_measured_time_raises_no_doubt_and_memory_is_the_region_s(self) -> None:
        best = {"wall_value": 1.0, "region_peak_rss_mb": 100, "peak_rss_mb": 1000}
        self.assertIsNone(_eval_ratchet._timing_warning("accept", "time_s", best, 2.0, measured=True))
        self.assertIsNotNone(_eval_ratchet._timing_warning("accept", "time_s", best, 2.0))
        warned = _eval_ratchet._memory_warning("accept", "time_s", best,
                                               {"region_peak_rss_mb": 200, "peak_rss_mb": 1100})
        self.assertIn("region_peak_rss_mb", warned)

    def test_results_say_when_the_harness_is_counted(self) -> None:
        unmarked = {"results": [{"metrics": {"time_s": 1.0}}]}
        marked = {"results": [{"metrics": {"time_s": 1.0, "regions": 1}}]}
        self.assertIn("mimir_region", eval_observe.harness_counted(unmarked))
        self.assertEqual(eval_observe.harness_counted(marked), "")


def _resolver(code=("src",), harness=("bench",)):
    root = "/ws"
    files = ["src/k.c", "src/k.py", "bench/h.py", "bench/h.c", "lib/util.py"]
    return frames.resolver(root, root, files, [f"{root}/{c}" for c in code], [f"{root}/{h}" for h in harness])


class HarnessApartTests(unittest.TestCase):
    """A stack through the harness and not through the code is the harness's."""

    def test_the_harness_s_work_is_counted_apart(self) -> None:
        stacks = [
            ([Frame("/ws/bench/h.py", "main"), Frame("/ws/src/k.py", "solve")], 50),       # the code
            ([Frame("/ws/bench/h.py", "main"), Frame("/ws/src/k.py", "solve"),
              Frame("/usr/lib/python3/site-packages/numpy/core.py", "dot")], 20),         # the code's numpy
            ([Frame("/ws/bench/h.py", "make_inputs"),
              Frame("/usr/lib/python3/site-packages/numpy/random.py", "rand")], 20),      # the harness's numpy
            ([Frame("/ws/bench/h.py", "check")], 5),                                         # the harness
            ([Frame("/usr/lib/libgomp.so.1", "gomp_thread_start")], 5),                      # a pool: kept
        ]
        prof = profiling.attribute(stacks, _resolver())
        self.assertEqual(prof["harness"], 0.25)
        self.assertEqual(prof["files"], {"src/k.py": round(70 / 75, 4)})
        self.assertEqual(prof["outside"], {"gomp": round(5 / 75, 4)})
        self.assertEqual(prof["functions"]["src/k.py::solve"]["self"], round(70 / 75, 4))

    def test_without_roles_everything_counts_as_before(self) -> None:
        stacks = [([Frame("/ws/bench/h.py", "check")], 5), ([Frame("/ws/src/k.py", "solve")], 5)]
        prof = profiling.attribute(stacks, frames.resolver("/ws", "/ws", ["src/k.py", "bench/h.py"]))
        self.assertNotIn("harness", prof)
        self.assertEqual(prof["files"], {"bench/h.py": 0.5, "src/k.py": 0.5})

    def test_roles(self) -> None:
        role = _resolver().role
        self.assertEqual([role(p) for p in ("src/k.c", "bench/h.py", "lib/util.py")], ["code", "harness", ""])
        # A harness named inside the code's directory is the harness still.
        role = frames.roles("/ws", ["/ws/src"], ["/ws/src/bench.py"])
        self.assertEqual((role("src/bench.py"), role("src/k.py")), ("harness", "code"))

    def test_the_runner_s_harness_is_its_file_and_its_own_folder(self) -> None:
        from mimir.servers.proxy import _proxy_runner
        cfg = {"optimize_paths": ["/ws/src/k.py"], "proxy_source_path": "/ws/proxy_bench/harnesses/h.py"}
        with mock.patch.object(_proxy_runner.store, "load_opt_config", return_value=cfg):
            self.assertEqual(_proxy_runner._roles("p", "/ws"),
                             (["/ws/src/k.py"], ["/ws/proxy_bench/harnesses/h.py", "/ws/proxy_bench/harnesses"]))
        cfg = {"optimize_paths": ["/ws/k.py"], "proxy_source_path": "/ws/h.py"}
        with mock.patch.object(_proxy_runner.store, "load_opt_config", return_value=cfg):
            self.assertEqual(_proxy_runner._roles("p", "/ws"), (["/ws/k.py"], ["/ws/h.py"]))


class TargetTests(_Tmp):
    """What runs, under whatever launches it."""

    def test_launchers_and_wrappers_are_not_the_program(self) -> None:
        cases = {
            ("mpirun", "-n", "2", "--mca", "btl", "self", "python3", "x.py", "-v"):
                (["mpirun", "-n", "2", "--mca", "btl", "self"], [], ["python3", "x.py", "-v"]),
            ("srun", "-n4", "--mpi=pmix", "./solver", "in"): (["srun", "-n4", "--mpi=pmix"], [], ["./solver", "in"]),
            ("env", "A=1", "uv", "run", "--with", "np", "python", "-m", "pkg"):
                ([], ["env", "A=1", "uv", "run", "--with", "np"], ["python", "-m", "pkg"]),
            ("taskset", "-c", "0-3", "./a.out"): ([], ["taskset", "-c", "0-3"], ["./a.out"]),
        }
        for argv, parts in cases.items():
            self.assertEqual(split(list(argv)), parts, argv)

    def test_what_runs_comes_from_the_measured_run_and_the_code(self) -> None:
        t = describe(["./driver"], {"programs": ["driver", "python3.12"], "runtimes": ["python3.12", "gfortran"]})
        self.assertTrue(t.python)
        self.assertEqual(t.native, {"fortran"})
        t = describe(["python3", "h.py"], {"programs": ["python3"], "runtimes": ["python3.10", "gfortran"]}, {"python"})
        self.assertEqual((t.python, t.native), (True, set()))
        t = describe(["python3", "h.py"], None, {"python", "c"})
        self.assertEqual((t.python, t.native, t.script), (True, {"c"}, "h.py"))
        self.assertTrue(describe(["bash", "-c", "python3 run.py"]).python)
        self.assertEqual(describe(["./a.out"], {"programs": ["a.out"], "runtimes": ["cudart"]}).gpu, "nvidia")

    def test_a_profiler_goes_in_front_of_rank_0_only(self) -> None:
        t = describe(["mpirun", "-n", "2", "python3", "-c", "import sys; print(sys.argv)", "a"])
        words = t.wrap(["echo", "PROFILED", "--"], self.dir)
        self.assertEqual(words[:3], ["mpirun", "-n", "2"])
        script = words[3:]
        rank0 = subprocess.run(script, capture_output=True, text=True, env={"PATH": os.environ["PATH"]}).stdout
        rank1 = subprocess.run(script, capture_output=True, text=True,
                               env={"PATH": os.environ["PATH"], "PMI_RANK": "1"}).stdout
        self.assertTrue(rank0.startswith("PROFILED -- python3"))
        self.assertEqual(rank1.strip(), "['-c', 'a']")
        # The interpreter's own profiler keeps the interpreter on the other ranks.
        script = t.wrap(["python3", "-m", "cProfile", "-o", os.devnull], self.dir, inside=True)[3:]
        rank1 = subprocess.run(script, capture_output=True, text=True,
                               env={"PATH": os.environ["PATH"], "SLURM_PROCID": "3"}).stdout
        self.assertEqual(rank1.strip(), "['-c', 'a']")

    def test_python_with_native_code_takes_perf_without_py_spy(self) -> None:
        with mock.patch.object(cpu.shutil, "which", side_effect=lambda t: None if t == "py-spy" else "/x"), \
             mock.patch.object(cpu, "perf_allowed", return_value=(True, "")):
            self.assertEqual(cpu.choose(describe(["python3", "h.py"], None, {"fortran"})), ("perf", ""))
            self.assertEqual(cpu.choose(describe(["python3", "h.py"], None, {"python"})), ("cprofile", ""))


class ReaderTests(unittest.TestCase):

    def test_perf_names_python_s_frames_from_3_12(self) -> None:
        text = ("\t7f00 _PyEval_EvalFrameDefault+0x10 (/usr/bin/python3.12)\n"
                "\t7f10 py::solve:/ws/src/k.py+0x5 ([unknown])\n"
                "\t7f20 py::main:/ws/bench/h.py+0x5 ([unknown])\n\n")
        frames_ = cpu.perf_stacks(text)[0][0]
        self.assertEqual(frames_[0], Frame("/ws/bench/h.py", "main"))
        self.assertEqual(frames_[1], Frame("/ws/src/k.py", "solve"))

    def test_callgrind_gives_a_library_s_count_to_the_code_that_calls_it(self) -> None:
        text = textwrap.dedent("""\
            events: Ir
            ob=/ws/h
            fl=/ws/bench/h.c
            fn=main
            5 10
            cfn=setup
            calls=1 7
            6 300
            cfn=work
            calls=1 3
            7 600
            fn=setup
            9 100
            cob=/usr/lib/libc.so.6
            cfl=???
            cfn=rand
            calls=10 0
            9 200
            ob=/ws/h
            fl=/ws/src/k.c
            fn=work
            3 500
            cob=/usr/lib/libc.so.6
            cfl=???
            cfn=memcpy
            calls=1 0
            4 100
            ob=/usr/lib/libc.so.6
            fl=???
            fn=rand
            0 150
            cfn=random_r
            calls=10 0
            0 50
            fn=random_r
            0 50
            fn=memcpy
            0 100
            """)
        own, calls, home = callgrind.read(text)
        stacks, inclusive = callgrind.stacks(own, calls, home, _resolver())
        prof = profiling.attribute(stacks, _resolver())
        # rand and random_r, under the harness's setup, are the harness's; memcpy is work's.
        self.assertEqual(prof["files"], {"src/k.c": 1.0})
        self.assertEqual(prof["functions"]["src/k.c::work"]["self"], 1.0)
        self.assertEqual(prof["harness"], round(310 / 910, 4))
        self.assertEqual(inclusive["work"], 600)


class CallgrindRunTests(_Tmp):
    """Where perf may not run, callgrind profiles a compiled run, its region alone."""

    def test_the_region_alone(self) -> None:
        for tool in ("gcc", "valgrind", "callgrind_control"):
            if not shutil.which(tool):
                self.skipTest(f"{tool} is not installed")
        self._write("src/k.c", "double work(double *a, int n){ double s=0; for(int r=0;r<40;r++) "
                               "for(int i=0;i<n;i++) s+=a[i]*a[i]; return s;}\n")
        self._write("bench/h.c", textwrap.dedent("""\
            #include <stdlib.h>
            #include <stdio.h>
            #include "mimir_region.h"
            double work(double*, int);
            static void setup(double *a, int n){ for(int k=0;k<30;k++) for(int i=0;i<n;i++) a[i]=rand(); }
            int main(){ int n=100000; double *a=malloc(n*8); setup(a,n);
              mimir_region_begin(); double s=work(a,n); mimir_region_end();
              setup(a,n); printf("%f\\n", s); return 0; }
            """))
        exe = os.path.join(self.dir, "h")
        subprocess.run(["gcc", "-g", "-O1", "-I", helpers.HERE, "bench/h.c", "src/k.c", "-o", exe],
                       cwd=self.dir, check=True)
        out = os.path.join(self.dir, "profile")
        os.makedirs(out)
        with mock.patch.object(cpu, "perf_allowed", return_value=(False, "perf is blocked here")):
            for regions in (True, False):
                prof = profiling.profile([exe], out, self.dir, ["src/k.c", "bench/h.c"], timeout=300,
                                         code=[os.path.join(self.dir, "src")],
                                         harness=[os.path.join(self.dir, "bench")], regions=regions)
                self.assertEqual(prof["tool"], "callgrind")
                self.assertGreater(prof["files"]["src/k.c"], 0.95, prof)
                if regions:
                    self.assertEqual(prof["window"], "measured region")
                    self.assertLess(prof.get("harness", 0), 0.01)
                else:
                    self.assertGreater(prof["harness"], 0.5)


_FAKE_CUDA = """\
#include <stdio.h>
#include <stdlib.h>
static void say(const char *w) { FILE *f = fopen(getenv("FAKE_CUDA_LOG"), "a"); fprintf(f, "%s\\n", w); fclose(f); }
int cuCtxSynchronize(void) { say("sync"); return 0; }
int cuProfilerStart(void) { say("start"); return 0; }
int cuProfilerStop(void) { say("stop"); return 0; }
int cuDeviceGetCount(int *n) { *n = 2; return 0; }
int cuDeviceGet(int *d, int i) { *d = i; return 0; }
int cuDevicePrimaryCtxGetState(int d, unsigned *f, int *a) { *f = 0; *a = d == 1; return 0; }
int cuDevicePrimaryCtxRetain(void **c, int d) { *c = (void *)(long)(d + 1); say("retain"); return 0; }
int cuDevicePrimaryCtxRelease_v2(int d) { (void)d; say("release"); return 0; }
int cuCtxPushCurrent_v2(void *c) { (void)c; say("push"); return 0; }
int cuCtxPopCurrent_v2(void **c) { (void)c; say("pop"); return 0; }
"""
_C_GPU = """\
#include <dlfcn.h>
#include <stdio.h>
#include "mimir_region.h"
int main(int argc, char **argv) {
    (void)argc;
    dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);       /* as a CUDA runtime loads the driver */
    mimir_region_begin(); puts("work"); mimir_region_end();
    return 0;
}
"""
_PY_GPU = """\
import ctypes, sys
sys.path.insert(0, sys.argv[1])
ctypes.CDLL(sys.argv[2])
from mimir_region import region
with region():
    pass
"""
# Each mark waits for the GPU (this thread's context, then device 1's, the active one),
# the region is the profiler's range, and the clocks are read inside it.
_GPU_CALLS = ["sync", "retain", "push", "sync", "pop", "release", "start",
              "sync", "retain", "push", "sync", "pop", "release", "stop"]


class GpuTests(_Tmp):
    """A GPU's work is asynchronous: the region waits for it, is the profiler's range,
    and weighs the GPU memory its processes hold."""

    def _libcuda(self) -> str:
        if not shutil.which("gcc"):
            self.skipTest("gcc is not installed")
        lib = os.path.join(self.dir, "libcuda.so.1")
        subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,-soname,libcuda.so.1", "-o", lib,
                        self._write("fake.c", _FAKE_CUDA)], check=True)
        return lib

    def _calls(self, argv: list[str]) -> list[str]:
        log = os.path.join(self.dir, "calls")
        open(log, "w").close()
        with open(os.devnull, "w") as out:
            done = proc_run.run(argv, rusage=True, timeout=60, stdout=out, env={**os.environ, "FAKE_CUDA_LOG": log})
        self.assertEqual(done.region["count"], 1)
        with open(log) as fh:
            return fh.read().split()

    def test_the_c_and_c_plus_plus_helper_waits_for_cuda(self) -> None:
        lib = self._libcuda()
        for compiler, flags in (("gcc", ["-std=c99"]), ("g++", ["-x", "c++"])):
            if not shutil.which(compiler):
                continue
            exe = os.path.join(self.dir, "h_" + compiler.replace("+", "p"))
            # The harness calls dlopen itself, which lives in libdl before glibc 2.34.
            subprocess.run([compiler, "-Wall", "-Wextra", "-Werror", *flags, "-I", helpers.HERE,
                            self._write("h.c", _C_GPU), "-o", exe, "-ldl"], check=True)
            self.assertEqual(self._calls([exe, lib]), _GPU_CALLS, compiler)

    def test_the_python_helper_waits_for_cuda(self) -> None:
        lib = self._libcuda()
        self.assertEqual(self._calls([sys.executable, self._write("h.py", _PY_GPU), helpers.HERE, lib]), _GPU_CALLS)

    def test_without_cuda_nothing_is_called(self) -> None:
        done = self._measure([sys.executable, self._write("h.py", _PY), helpers.HERE])
        self.assertNotIn("device_peak_mb", done.region)

    def test_the_region_weighs_the_gpu_memory_of_its_own_processes(self) -> None:
        from mimir.servers.proxy._lib import region as reader
        seen = iter([{"42": 300.0, "7": 9000.0}, {"42": 900.0, "7": 9000.0}] + [{"42": 400.0}] * 50)
        with mock.patch.object(reader, "sample_device", side_effect=lambda vendor: next(seen)), \
             mock.patch.object(reader, "_clear_peak", return_value=True):
            r = reader.Regions(lambda: ["42"], gpu=lambda: "nvidia")
            r._mark("begin", 0.0)
            import time
            time.sleep(0.3)
            r._mark("end", 1.0)
        self.assertEqual((r.summary()["device_peak_mb"], r.summary()["device_scope"]), (900.0, "process"))
        m = {}
        metrics.normalize_time_metrics(m, 2.0, r.summary())
        self.assertEqual(m["region_peak_device_mb"], 900.0)

    def _fake_nsys(self, captures: bool) -> None:
        """An nsys whose report holds _NSYS_KERN; under --capture-range, only when the run
        was given MIMIR_REGION (its helpers open the range) and *captures*."""
        from mimir.tests.test_profiling import _NSYS_KERN
        kern = self._write("kern.csv", _NSYS_KERN)
        bindir = os.path.join(self.dir, "bin")
        os.makedirs(bindir)
        self._write("bin/nsys", textwrap.dedent(f"""\
            #!{sys.executable}
            import os, shutil, sys
            args = sys.argv[1:]
            if args[0] == "profile":
                rep = args[args.index("-o") + 1]
                ranged = "--capture-range=cudaProfilerApi" in args
                with open({os.path.join(self.dir, "nsys.calls")!r}, "a") as fh:
                    fh.write(("ranged" if ranged else "whole") + " " + str("MIMIR_REGION" in os.environ) + "\\n")
                if not ranged or ({captures!r} and "MIMIR_REGION" in os.environ):
                    open(rep + ".nsys-rep", "w").close()
            else:
                shutil.copy({kern!r}, args[args.index("--output") + 1] + "_cuda_gpu_kern_sum.csv")
            """))
        os.chmod(os.path.join(bindir, "nsys"), 0o755)
        patcher = mock.patch.dict(os.environ, {"PATH": bindir + os.pathsep + os.environ["PATH"]})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _trace(self, region_s):
        from mimir.servers.proxy._lib.profiling import gpu
        with mock.patch.object(gpu, "sample_device", return_value=None), open(os.devnull, "w") as log:
            return gpu.trace("nsys", describe(["true"]), self.dir, self.dir, [], log, 60, 20.0, region_s)

    def test_nsys_traces_the_region_alone(self) -> None:
        self._fake_nsys(captures=True)
        out = self._trace(10.0)
        self.assertEqual(out["window"], "measured region")
        self.assertEqual(out["busy"], 1.0)              # 10 s of kernels in a 10 s region
        with open(os.path.join(self.dir, "nsys.calls")) as fh:
            self.assertEqual(fh.read().split(), ["ranged", "True"])

    def test_a_region_nsys_could_not_trace_alone_is_traced_whole(self) -> None:
        self._fake_nsys(captures=False)
        out = self._trace(10.0)
        self.assertNotIn("window", out)
        self.assertEqual(out["busy"], 0.5)              # against the run's 20 s
        self.assertIn("whole run", out["note"])
        with open(os.path.join(self.dir, "nsys.calls")) as fh:
            self.assertEqual(fh.read().split(), ["ranged", "True", "whole", "False"])

    def test_without_a_region_nsys_traces_the_run(self) -> None:
        self._fake_nsys(captures=True)
        out = self._trace(None)
        self.assertNotIn("window", out)
        with open(os.path.join(self.dir, "nsys.calls")) as fh:
            self.assertEqual(fh.read().split(), ["whole", "False"])


class SessionTests(_TmpStorageTest):
    """Through the runner: the objective is the region's time, the profile the code's."""

    def test_a_marked_harness_through_a_session(self) -> None:
        helpers.install(os.path.join(self.root, "bench"))
        exe = os.path.join(self.root, "bench", "h.py")
        with open(exe, "w") as fh:
            fh.write(textwrap.dedent("""                import os, sys, time
                import numpy as np
                here = os.path.dirname(os.path.abspath(__file__))
                sys.path[:0] = [here, os.path.dirname(here)]
                from mimir_region import region
                import tracked
                junk = sum(i * i for i in range(600000))          # the harness's own work
                with region():
                    total = tracked.work()
                out = sys.argv[1] if sys.argv[1].endswith('.npz') else sys.argv[1] + '.npz'
                np.savez(out, field=np.ones(4))
                print('PROXY_METRICS_BEGIN')
                print('time_s=9.0')
                print(f'output_file={out}')
                print('PROXY_METRICS_END')
                """))
        tracked = os.path.join(self.root, "tracked.py")
        with open(tracked, "w") as fh:
            fh.write("def work():\n    return sum(i * i for i in range(300000))\n")
        server_proxy.proxy_manage(op="register", name="m", executable_path=exe,
                                  run_cmd_template=f"{sys.executable} {{executable}} {{output_file}}",
                                  output_format="npz", confirm=True)
        server_proxy.proxy_manage(op="suite_define", name="mb", cases=[{"case_id": "a", "proxy_name": "m"}],
                                  confirm=True)
        res = _eval(op="init", proxy_name="m", benchmark_name="mb", proxy_source_path=exe,
                    optimize_paths=[tracked], primary_metric="time_s",
                    requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}], confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        real_which = shutil.which
        with mock.patch.object(cpu.shutil, "which", side_effect=lambda t: None if t == "py-spy" else real_which(t)):
            cfg, error, run_dir, _ = eval_session.prepare_run("m", "", False)
            self.assertIsNone(error)
            procs.update_opt_active_link("m", run_dir)
            from mimir.servers.proxy import _proxy_runner
            with mock.patch.object(sys, "argv", ["_proxy_runner", "--run-dir", run_dir]):
                _proxy_runner.main()
        case = read_json(os.path.join(run_dir, "metrics.json"))["results"][0]["metrics"]
        self.assertEqual(case["regions"], 1)
        self.assertEqual(case["time_s"], case["region_time_s"])
        self.assertEqual(case["time_s_ignored"], 9.0)
        self.assertLess(case["region_time_s"], case["wall_time_s"])
        prof = read_json(profiling.profile_path(run_dir))
        self.assertEqual(prof["tool"], "cprofile")
        self.assertIn("tracked.py", prof["files"])
        self.assertNotIn("bench/h.py", prof["files"])
        # Twice tracked's work; a fixed share would depend on how long NumPy takes to import.
        self.assertGreater(prof["harness"], prof["files"]["tracked.py"])


if __name__ == "__main__":
    unittest.main()
