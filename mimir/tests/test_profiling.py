"""Where a run's time goes, by workspace file, function and line: the profilers' outputs
read back, each sample given to the deepest workspace frame of its stack, the profiler that
fits chosen (or why none does), the view the agent asks for, and the runner profiling after
the verdict without touching it."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from mimir.servers.proxy._lib import procs, profiling
from mimir.servers.proxy._lib.profiling import Frame, cpu, frames, gpu, memory
from mimir.servers.proxy import server_proxy
from mimir.servers.proxy._ops import eval_session
from mimir.tests.support.proxy import _TmpStorageTest, _eval
from mimir.shared.atomic import read_json

_PYSPY = """\
0x7f (libc.so.6);<module> (main.py:3);heavy (pkg/work.py:5);0x7e (libm.so.6) 3
0x7f (libc.so.6);<module> (main.py:3);heavy (pkg/work.py:5) 5
0x7f (libc.so.6);<module> (main.py:3);light (pkg/work.py:8) 1
0x7f (libc.so.6);_bootstrap (/usr/lib/python3.12/site-packages/numpy/core/x.py:1) 1
"""

_PERF = """\
\t    55d0c3a4b129 kernel+0x19 (/ws/build/solver)
  /ws/src/kernel.f90:12
\t    55d0c3a4b170 main+0x20 (/ws/build/solver)
  /ws/src/main.c:20

\t    7f3 __memcpy_avx (/usr/lib/libc.so.6)
  ??:0
\t    55d0c3a4b170 main+0x20 (/ws/build/solver)
  /ws/src/main.c:21

\t    7f3 cblas_dgemm (/usr/lib/libopenblas.so.0)
\t    7f4 start (/usr/lib/libc.so.6)
"""


def _resolve(known: dict[str, str]):
    return lambda path: known.get(path)


class ReadingTests(unittest.TestCase):
    def test_py_spy_stacks_keep_their_files_and_libraries_root_first(self) -> None:
        stacks = profiling.pyspy_stacks(_PYSPY)
        self.assertEqual(stacks[0], ([Frame("libc.so.6"), Frame("main.py", "<module>", 3),
                                      Frame("pkg/work.py", "heavy", 5), Frame("libm.so.6")], 3))

    def test_a_sample_goes_to_its_deepest_workspace_frame(self) -> None:
        resolve = _resolve({"main.py": "main.py", "pkg/work.py": "pkg/work.py"})
        prof = profiling.attribute(profiling.pyspy_stacks(_PYSPY), resolve)
        # libm called from heavy() is heavy's time; numpy's import is outside.
        self.assertEqual(prof["samples"], 10)
        self.assertEqual(prof["files"], {"pkg/work.py": 0.9})
        self.assertEqual(prof["outside"], {"numpy": 0.1})

    def test_functions_keep_their_own_time_and_the_time_under_them(self) -> None:
        resolve = _resolve({"main.py": "main.py", "pkg/work.py": "pkg/work.py"})
        prof = profiling.attribute(profiling.pyspy_stacks(_PYSPY), resolve)
        self.assertEqual(prof["functions"], {
            "pkg/work.py::heavy": {"self": 0.8, "total": 0.8, "line": 5},
            "pkg/work.py::light": {"self": 0.1, "total": 0.1, "line": 8},
            "main.py::<module>": {"self": 0.0, "total": 0.9},
        })
        self.assertEqual(prof["lines"], {"pkg/work.py:5": 0.8, "pkg/work.py:8": 0.1})

    def test_perf_script_frames_use_the_source_file_when_perf_knows_it(self) -> None:
        stacks = profiling.perf_stacks(_PERF)
        self.assertEqual(stacks[0][0], [Frame("/ws/src/main.c", "main", 20), Frame("/ws/src/kernel.f90", "kernel", 12)])
        self.assertEqual([f.where for f in stacks[1][0]], ["/ws/src/main.c", "/usr/lib/libc.so.6"])
        resolve = _resolve({"/ws/src/main.c": "src/main.c", "/ws/src/kernel.f90": "src/kernel.f90"})
        prof = profiling.attribute(stacks, resolve)
        self.assertEqual(prof["files"], {"src/kernel.f90": 0.3333, "src/main.c": 0.3333})
        self.assertEqual(prof["outside"], {"openblas": 0.3333})
        self.assertEqual(prof["functions"]["src/main.c::main"], {"self": 0.3333, "total": 0.6667, "line": 21})
        self.assertEqual(prof["lines"], {"src/kernel.f90:12": 0.3333, "src/main.c:21": 0.3333})

    def test_dram_traffic_from_the_controllers_else_from_cache_misses(self) -> None:
        core = "1000000000,,LLC-loads:u,100,100.00,,\n250000000,,LLC-load-misses:u,100,100.00,,\n"
        uncore = ("4096.00,MiB,uncore_imc_0/cas_count_read/,100,100.00,,\n"
                  "4096.00,MiB,uncore_imc_1/cas_count_read/,100,100.00,,\n"
                  "2048.00,MiB,uncore_imc/cas_count_write/,100,100.00,,\n")
        found = cpu.bandwidth(core, uncore, 2.0, 50.0)
        self.assertEqual(found["dram_gbytes_per_s"], round(10240 * 2**20 / 2.0 / 1e9, 2))
        self.assertEqual(found["bandwidth_fraction"], round(10240 * 2**20 / 2.0 / 1e9 / 50.0, 3))
        self.assertIn("controllers", found["dram_source"])
        estimate = cpu.bandwidth(core, "", 2.0, None)
        self.assertEqual(estimate["dram_gbytes_per_s"], 8.0)
        self.assertIn("lower bound", estimate["dram_source"])
        self.assertNotIn("bandwidth_fraction", estimate)
        self.assertEqual(cpu.perf_stat(core)["llc_miss_rate"], 0.25)

    def test_fortran_symbols_read_as_the_source_names_them(self) -> None:
        self.assertEqual(frames.fortran_name("__solver_MOD_relax"), "solver::relax")
        self.assertEqual(frames.fortran_name("solver_mp_relax_"), "solver::relax")
        self.assertEqual(frames.fortran_name("smooth_", "src/a.f90"), "smooth")
        self.assertEqual(frames.fortran_name("smooth_", "src/a.c"), "smooth_")
        self.assertEqual(frames.fortran_name("MAIN__"), "main program")
        stacks = profiling.perf_stacks(_PERF.replace("kernel+0x19", "__solver_MOD_relax+0x19"))
        self.assertEqual(stacks[0][0][-1].func, "solver::relax")

    def test_stacks_cut_short_are_unwound_again_from_the_stack(self) -> None:
        cut = [([Frame("/ws/build/solver", "relax")], 9), ([Frame("/ws/src/main.c", "main"), Frame("x")], 1)]
        whole = [([Frame("/ws/src/main.c", "main"), Frame("/ws/src/k.c", "relax")], 10)]
        calls = []

        def perf(argv, out_dir, log, timeout, unwind):
            calls.append(unwind)
            return cut if unwind == "fp" else whole
        with mock.patch.object(cpu, "_perf", side_effect=perf):
            prof = cpu.sample("perf", ["./solver"], "/tmp", lambda p: p if p.startswith("/ws/src") else None,
                              None, None)
        self.assertEqual(calls, ["fp", "dwarf"])
        self.assertEqual(prof["functions"]["/ws/src/main.c::main"]["total"], 1.0)
        self.assertNotIn("note", prof)

    def test_a_binary_without_debug_information_is_named(self) -> None:
        bare = [([Frame("/ws/build/solver", "relax"), Frame("/ws/build/solver", "main")], 10)]
        with mock.patch.object(cpu, "_perf", return_value=bare):
            prof = cpu.sample("perf", ["/ws/build/solver"], "/tmp", lambda p: None, None, None)
        self.assertIn("add -g", prof["note"])

    def test_py_spy_s_process_frames_are_not_frames(self) -> None:
        text = ('process 7:"python3 h.py";process 8:"python3 -c a;b (x)";'
                '<module> (/ws/src/k.py:9);work (/ws/src/k.py:3) 20\n')
        self.assertEqual(profiling.pyspy_stacks(text),
                         [([Frame("/ws/src/k.py", "<module>", 9), Frame("/ws/src/k.py", "work", 3)], 20)])

    def _harness_waiting(self, perf_ok: tuple[bool, str], native: list | None = None) -> dict:
        """A Python harness whose code runs in a compiled child: py-spy sees it wait."""
        resolve = lambda p: p.removeprefix("/ws/") if p.startswith("/ws/") else None
        waiting = [([Frame("/ws/bench/h.py", "<module>", 2), Frame("libc.so.6", "wait4")], 5)]
        with mock.patch.object(cpu, "_pyspy", return_value=profiling.attribute(waiting, resolve)), \
             mock.patch.object(cpu, "perf_allowed", return_value=perf_ok), \
             mock.patch.object(cpu, "_perf", return_value=native or []):
            return cpu.sample("py-spy", ["python3", "h.py"], "/tmp", resolve, None, None, wall_s=2.0)

    def test_a_harness_seen_waiting_on_a_compiled_child_is_profiled_by_perf(self) -> None:
        native = [([Frame("/usr/bin/python3.12"), Frame("/ws/src/main.c", "main", 20)], 1)] * 900
        prof = self._harness_waiting((True, ""), native)
        self.assertEqual((prof["tool"], prof["files"]), ("perf", {"src/main.c": 1.0}))
        self.assertIn("py-spy saw 0.0 s of the run's 2.0 s", prof["note"])
        self.assertNotIn("debug information", prof["note"])

    def test_a_harness_seen_waiting_says_why_perf_did_not_take_over(self) -> None:
        prof = self._harness_waiting((False, "perf is not installed"))
        self.assertEqual(prof["files"], {"bench/h.py": 1.0})
        self.assertIn("perf would see it, but perf is not installed", prof["note"])
        prof = self._harness_waiting((True, ""), [([Frame("/usr/bin/python3.12")], 1)] * 900)
        self.assertEqual(prof["files"], {"bench/h.py": 1.0})
        self.assertIn("perf saw no more of the workspace", prof["note"])

    def test_a_py_spy_profile_that_saw_the_run_is_kept(self) -> None:
        seen = {"samples": 400, "files": {"src/k.py": 1.0}, "outside": {}}
        with mock.patch.object(cpu, "_pyspy", return_value=seen), mock.patch.object(cpu, "_perf") as perf:
            self.assertEqual(cpu.sample("py-spy", ["python3", "h.py"], "/tmp", None, None, None, wall_s=2.0), seen)
            self.assertEqual(cpu.sample("py-spy", ["python3", "h.py"], "/tmp", None, None, None), seen)
        perf.assert_not_called()

    def test_perf_stat_gives_instructions_per_cycle_and_miss_rates(self) -> None:
        text = ("# started on Mon\n\n2000,,cycles:u,100,100.00,,\n3000,,instructions:u,100,100.00,1.50,insn per cycle\n"
                "100,,cache-references:u,100,100.00,,\n25,,cache-misses:u,100,100.00,,\n"
                "<not supported>,,branches:u,0,100.00,,\n7,,branch-misses:u,100,100.00,,\n")
        self.assertEqual(cpu.perf_stat(text), {"ipc": 1.5, "cache_miss_rate": 0.25})

    def test_frames_resolve_by_absolute_path_by_cwd_and_by_unique_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            files = ["src/a.py", "lib/k.cpp", "x/util.h", "y/util.h"]
            resolve = profiling.resolver(root, os.path.join(root, "src"), files)
            self.assertEqual(resolve(os.path.join(root, "src", "a.py")), "src/a.py")
            self.assertEqual(resolve("a.py"), "src/a.py")
            self.assertEqual(resolve("/build/tree/lib/k.cpp"), "lib/k.cpp")
            self.assertIsNone(resolve("util.h"))
            self.assertIsNone(resolve("/usr/lib/python3/os.py"))


class ChoiceTests(unittest.TestCase):
    def test_python_prefers_py_spy_then_cprofile(self) -> None:
        with mock.patch.object(cpu.shutil, "which", return_value="/bin/py-spy"):
            self.assertEqual(profiling.choose(["python3", "run.py"]), ("py-spy", ""))
        with mock.patch.object(cpu.shutil, "which", return_value=None):
            self.assertEqual(profiling.choose(["python3", "run.py"]), ("cprofile", ""))
            tool, why = profiling.choose(["python3", "-c", "x"])
            self.assertEqual(tool, "")
            self.assertIn("cProfile", why)

    def test_a_compiled_run_needs_perf_and_says_how_to_allow_it(self) -> None:
        with mock.patch.object(cpu, "perf_allowed", return_value=(True, "")):
            self.assertEqual(profiling.choose(["./solver"]), ("perf", ""))
        with mock.patch.object(cpu.shutil, "which", return_value="/usr/bin/perf"), \
             mock.patch.object(cpu.os, "geteuid", return_value=1000), \
             mock.patch("builtins.open", mock.mock_open(read_data="4\n")):
            tool, why = profiling.choose(["./solver"])
        self.assertEqual(tool, "")
        self.assertIn("perf_event_paranoid = 4", why)
        self.assertIn("sysctl", why)


_PROFILE = {
    "tool": "py-spy", "case": "a", "samples": 10,
    "files": {"pkg/work.py": 0.9, "main.py": 0.05},
    "functions": {"pkg/work.py::heavy": {"self": 0.8, "total": 0.8},
                  "pkg/work.py::light": {"self": 0.1, "total": 0.1},
                  "main.py::<module>": {"self": 0.05, "total": 0.95}},
    "lines": {"pkg/work.py:5": 0.8, "pkg/work.py:8": 0.1, "main.py:3": 0.05},
    "outside": {"numpy": 0.05},
}


class ViewTests(unittest.TestCase):
    """What the agent reads at each level."""

    def test_the_file_view_is_the_costliest_files_and_the_share_outside(self) -> None:
        self.assertEqual(profiling.top(_PROFILE, 1),
                         {"tool": "py-spy", "case": "a", "files": {"pkg/work.py": 0.9}, "outside": 0.05})

    def test_the_function_and_line_views_narrow_to_the_file_in_focus(self) -> None:
        view = profiling.top(_PROFILE, 5, "function", "pkg/work.py")
        self.assertEqual(list(view["functions"]), ["pkg/work.py::heavy", "pkg/work.py::light"])
        self.assertEqual(profiling.top(_PROFILE, 1, "line")["lines"], {"pkg/work.py:5": 0.8})
        self.assertEqual(profiling.top(_PROFILE, 5, "line", "main.py")["lines"], {"main.py:3": 0.05})

    def test_a_level_the_profile_lacks_says_why(self) -> None:
        old = {k: v for k, v in _PROFILE.items() if k not in ("functions", "lines")}
        self.assertIn("profile again", profiling.top(old, level="function")["unavailable"])
        self.assertIn("py-spy", profiling.top({**old, "tool": "cprofile"}, level="line")["unavailable"])
        self.assertIn("level='counters'", profiling.top(_PROFILE, level="counters")["unavailable"])
        self.assertFalse(profiling.has(old, "line"))
        self.assertTrue(profiling.has(_PROFILE, "line"))

    def test_counters_are_for_compiled_code(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            prof = profiling.profile([sys.executable, "x.py"], out, out, [], level="counters")
        self.assertIn("compiled code", prof["counters"]["unavailable"])


# Reconstructed from the tools' documented CSV layouts, recent and older headers.
_NSYS_KERN = """\
"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
"70.0","7000000000","400","17500000.0","17500000.0","17000000","18000000","1000.0","void k::stencil_kernel<float>(float *, const float *, int)"
"20.0","2000000000","10","200000000.0","200000000.0","2","2","0.0","volta_sgemm_128x64_nn"
"10.0","1000000000","100","10000000.0","10000000.0","1","1","0.0","relax(float *)"
"""
_NSYS_MEM_OLD = """\
"Time(%)","Total Time (ns)","Operations","Average","Minimum","Maximum","StdDev","Operation"
"80.0","800000000","2","400000000.0","1","1","0.0","[CUDA memcpy HtoD]"
"20.0","200000000","1","200000000.0","1","1","0.0","[CUDA memcpy DtoH]"
"""
_ROCPROF_KERN = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"stencil_kernel(float*, float const*, int) [clone .kd]",400,6000000000,15000000,100.0,1,1,0
"""
_ROCPROF_MEM = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"MEMORY_COPY_HOST_TO_DEVICE",2,500000000,250000000,100.0,1,1,0
"""
_NCU = """\
==PROF== Connected to process 1234
"ID","Process ID","Process Name","Kernel Name","sm__throughput.avg.pct_of_peak_sustained_elapsed","dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active"
"","","","","%","%","%"
"0","1234","solver","stencil_kernel","34.10","88.52","50.00"
"""
_CU = """\
#include <cuda_runtime.h>
namespace k {
template <typename T>
__global__ void stencil_kernel(T* out, const T* in, int n) { out[0] = in[0]; }
}
__global__ void __launch_bounds__(256) relax(float* x) { x[0] = 0; }
void host_side(float* x) { relax<<<1, 1>>>(x); }
"""

_F90 = """\
module solver
contains
  subroutine relax(x, n)
    real :: x(n)
    !$acc parallel loop
    do i = 1, n
      x(i) = 0
    end do
  end subroutine
end module
subroutine smooth(x)
end subroutine
"""
_CUF = """\
module k
contains
  attributes(global) subroutine KERN(a)
    real :: a(*)
  end subroutine
end module
"""
_ACC_C = """\
#include <stdlib.h>
void saxpy(int n, float a, float *x, float *y)
{
  #pragma acc parallel loop
  for (int i = 0; i < n; i++) y[i] += a * x[i];
}
"""


def _write_files(root: str, files: dict[str, str]) -> None:
    for rel, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as fh:
            fh.write(text)


class GpuReadingTests(unittest.TestCase):
    """Kernels and copies as shares of the run's time, each kernel with its file."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_a_kernel_s_bare_name_and_the_file_that_defines_it(self) -> None:
        self.assertEqual(gpu.kernel_base("void k::stencil_kernel<float>(float *, const float *, int)"), "stencil_kernel")
        self.assertEqual(gpu.kernel_base("stencil_kernel(float*, float const*, int) [clone .kd]"), "stencil_kernel")
        self.assertEqual(gpu.kernel_base("volta_sgemm_128x64_nn"), "volta_sgemm_128x64_nn")
        _write_files(self.root, {"src/stencil.cu": _CU, "src/copy.cu": _CU.replace("relax", "other"),
                                 "src/host.c": "void relax(void) {}\n"})
        index = gpu.kernel_index(self.root, ["src/stencil.cu", "src/copy.cu", "src/host.c"])
        self.assertEqual(index["relax"], ["src/stencil.cu"])
        self.assertEqual(sorted(index["stencil_kernel"]), ["src/copy.cu", "src/stencil.cu"])

    def test_fortran_and_compiler_made_kernels_find_their_file_and_line(self) -> None:
        _write_files(self.root, {"src/solver.f90": _F90, "src/kern.cuf": _CUF, "src/host.c": _ACC_C})
        home = gpu.kernel_homes(self.root, ["src/solver.f90", "src/kern.cuf", "src/host.c"])
        self.assertEqual(home("kern"), ("src/kern.cuf", 0))
        self.assertEqual(home("relax_6_gpu"), ("src/solver.f90", 6))
        self.assertEqual(home("nvkernel_relax_F1L6_1"), ("src/solver.f90", 6))
        self.assertEqual(home("__omp_offloading_10302_2a0c8f1___solver_MOD_relax_l6"), ("src/solver.f90", 6))
        self.assertEqual(home("__omp_offloading_fd00_4c1_smooth__l3"), ("src/solver.f90", 3))
        self.assertEqual(home("saxpy_4_gpu"), ("src/host.c", 4))
        self.assertEqual(home("volta_sgemm_128x64_nn"), ("", 0))

    def test_nsys_kernels_and_copies_against_the_measured_time(self) -> None:
        prefix = os.path.join(self.root, "trace")
        _write_files(self.root, {"trace_cuda_gpu_kern_sum.csv": _NSYS_KERN,
                                 "trace_gpumemtimesum.csv": _NSYS_MEM_OLD, "src/stencil.cu": _CU})
        summary = gpu.nsys_summary(prefix)
        self.assertEqual(summary["memcpy"], {"HtoD": 800000000.0, "DtoH": 200000000.0})
        out = gpu.summarize(summary, 20.0, gpu.kernel_homes(self.root, ["src/stencil.cu"]))
        self.assertEqual(out["busy"], 0.5)
        self.assertEqual(out["memcpy"], {"HtoD": 0.04, "DtoH": 0.01})
        self.assertEqual(out["idle"], 0.45)
        # One file defines both workspace kernels; cuBLAS's is outside the workspace.
        self.assertEqual(out["files"], {"src/stencil.cu": 0.4})
        self.assertEqual(out["outside"], 0.1)
        first = out["kernels"]["void k::stencil_kernel<float>(float *, const float *, int)"]
        self.assertEqual(first, {"share": 0.35, "calls": 400, "file": "src/stencil.cu"})
        self.assertNotIn("file", out["kernels"]["volta_sgemm_128x64_nn"])

    def test_the_bytes_copied_whatever_unit_the_report_uses(self) -> None:
        new = '"Total (MB)","Count","Avg (MB)","Operation"\n"2048.000","2","1024.0","[CUDA memcpy Host-to-Device]"\n' \
              '"1.000","1","1.0","[CUDA memset]"\n'
        old = '"Total","Operations","Average","Operation"\n"1024.000","1","1024.0","[CUDA memcpy DtoH]"\n'
        _write_files(self.root, {"a_cuda_gpu_kern_sum.csv": _NSYS_KERN, "a_cuda_gpu_mem_size_sum.csv": new,
                                 "b_gpukernsum.csv": _NSYS_KERN, "b_gpumemsizesum.csv": old})
        self.assertEqual(gpu.nsys_summary(os.path.join(self.root, "a"))["memcpy_bytes"], {"HtoD": 2.048e9})
        self.assertEqual(gpu.nsys_summary(os.path.join(self.root, "b"))["memcpy_bytes"], {"DtoH": 2**20})

    def test_the_device_memory_of_the_run_s_own_processes(self) -> None:
        samples = iter([{"7": 500.0},                        # before: someone else's job
                        {"7": 500.0, "42": 1200.0}, {"7": 500.0, "42": 3000.0, "43": 100.0},
                        {"7": 500.0}])                       # after: the run is gone
        with mock.patch.object(gpu, "sample_device", side_effect=lambda _v: next(samples, {"7": 500.0})), \
             mock.patch.object(gpu.DeviceMemory, "INTERVAL_S", 0.01):
            device = gpu.DeviceMemory("nvidia")
            with device:
                time.sleep(0.1)
        self.assertEqual(device.peak_mb, 3100.0)

    def test_overlapping_kernels_are_capped_and_said(self) -> None:
        out = gpu.summarize({"kernels": {"a": (3e9, 1)}, "memcpy": {}}, 2.0, lambda _name: ("", 0))
        self.assertEqual((out["busy"], out["idle"]), (1.0, 0.0))
        self.assertIn("overlap", out["note"])

    def test_rocprof_statistics_are_summed_over_processes(self) -> None:
        for pid in ("101", "102"):
            _write_files(self.root, {f"gpu/host/{pid}/run_kernel_stats.csv": _ROCPROF_KERN,
                                     f"gpu/host/{pid}/run_memory_copy_stats.csv": _ROCPROF_MEM})
        summary = gpu.rocprof_summary(os.path.join(self.root, "gpu"))
        self.assertEqual(summary["kernels"], {"stencil_kernel(float*, float const*, int) [clone .kd]": (12e9, 800)})
        self.assertEqual(summary["memcpy"], {"HtoD": 1e9})

    def test_a_header_the_reader_does_not_know_is_said(self) -> None:
        _write_files(self.root, {"t_cuda_gpu_kern_sum.csv": '"Name","Duration"\n"k","1"\n'})
        with self.assertRaisesRegex(ValueError, "Total"):
            gpu.nsys_summary(os.path.join(self.root, "t"))

    def test_ncu_says_what_bounds_the_kernel(self) -> None:
        _write_files(self.root, {"ncu.csv": _NCU})
        found = gpu.ncu_detail(os.path.join(self.root, "ncu.csv"), "stencil_kernel")
        self.assertEqual(found, {"name": "stencil_kernel", "sm_pct": 34.1, "dram_pct": 88.52, "occupancy_pct": 50.0})
        self.assertEqual(gpu.bound(found), "memory")
        self.assertEqual(gpu.bound({"sm_pct": 80, "dram_pct": 30}), "compute")
        self.assertEqual(gpu.bound({"sm_pct": 10, "dram_pct": 12}), "latency")


class GpuChoiceTests(unittest.TestCase):
    def test_the_vendor_comes_from_the_libraries_then_the_job_s_devices(self) -> None:
        with mock.patch.object(gpu, "_linked", return_value="libcudart.so.12 => /usr/lib/libcudart.so.12"):
            self.assertEqual(gpu.gpu_vendor(["./solver"]), "nvidia")
        with mock.patch.object(gpu, "_linked", return_value=""), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gpu.gpu_vendor(["python3", "x.py"], ["amdhip64"]), "amd")
            self.assertEqual(gpu.gpu_vendor(["python3", "x.py"], ["numpy"]), "")
        with mock.patch.object(gpu, "_linked", return_value=""), \
             mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
             mock.patch.object(gpu.shutil, "which", return_value=None):
            self.assertEqual(gpu.gpu_vendor(["./solver"]), "nvidia")

    def test_a_missing_tracer_says_to_load_it(self) -> None:
        with mock.patch.object(gpu.shutil, "which", return_value=None):
            tool, why = gpu.choose_gpu("nvidia")
        self.assertEqual(tool, "")
        self.assertIn("nsys", why)
        self.assertIn("module", why)

    def test_a_run_on_a_gpu_carries_its_gpu_section(self) -> None:
        traced = {"wall_s": 2.0, "busy": 0.5, "memcpy": {}, "idle": 0.5, "files": {}, "kernels": {"k": {"share": 0.5, "calls": 1}}}
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.object(profiling, "choose", return_value=("", "no CPU profiler")), \
             mock.patch.object(gpu, "gpu_vendor", return_value="nvidia"), \
             mock.patch.object(gpu, "choose_gpu", return_value=("nsys", "")), \
             mock.patch.object(gpu, "trace", return_value=traced):
            prof = profiling.profile(["./solver"], out, out, [], level="file")
        self.assertEqual(prof["gpu"]["busy"], 0.5)
        # The CPU side is missing, so the view says so; the GPU side is still there.
        view = profiling.top(prof)
        self.assertEqual(view["unavailable"], "no CPU profiler")
        self.assertEqual(view["gpu"]["kernels"], {"k": {"share": 0.5, "calls": 1}})

    def test_the_file_view_carries_the_gpu_summary(self) -> None:
        prof = {**_PROFILE, "gpu": {"tool": "nsys", "busy": 0.5, "memcpy": {"HtoD": 0.1}, "idle": 0.4,
                                    "kernels": {"a": {"share": 0.3}, "b": {"share": 0.2}}}}
        view = profiling.top(prof, 1)["gpu"]
        self.assertEqual(view, {"tool": "nsys", "busy": 0.5, "memcpy": {"HtoD": 0.1}, "idle": 0.4,
                                "kernels": {"a": {"share": 0.3}}})

    def test_a_kernel_view_without_a_name_lists_the_known_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            _write_files(out, {"profile.json": json.dumps({"gpu": {"vendor": "nvidia", "kernels": {"stencil": {}}}})})
            prof = profiling.profile(["./solver"], out, out, [], level="kernel")
        self.assertIn("stencil", prof["kernel_detail"]["unavailable"])
        self.assertIn("files", profiling.top({**_PROFILE, **prof}))


_MASSIF = """\
desc: --threshold=0.5
cmd: ./solver
time_unit: i
#-----------
snapshot=0
#-----------
time=0
mem_heap_B=0
mem_heap_extra_B=0
mem_stacks_B=0
heap_tree=empty
#-----------
snapshot=1
#-----------
time=100
mem_heap_B=30000000
mem_heap_extra_B=8
mem_stacks_B=0
heap_tree=peak
n3: 30000000 (heap allocation functions) malloc/new/new[], --alloc-fns, etc.
 n1: 20000000 0x1091DC: __solver_MOD_relax (solver.f90:6)
  n1: 20000000 0x109320: MAIN__ (solver.f90:14)
   n0: 20000000 0x109320: main (solver.f90:13)
 n1: 9000000 0x4C2A: std::vector<double>::reserve(unsigned long) (in /usr/lib/libstdc++.so.6)
  n0: 9000000 0x109400: grid::build(int) (grid.cpp:42)
 n0: 1000000 in 7 places, all below massif's threshold (0.50%)
"""


class MemoryTests(unittest.TestCase):
    """The allocations live at the heap's peak, by the workspace frame that made them."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_massif_s_peak_goes_to_the_deepest_workspace_frame(self) -> None:
        peak, stacks = memory.massif_peak(_MASSIF)
        self.assertEqual(peak, 30000000)
        resolve = _resolve({"solver.f90": "src/solver.f90", "grid.cpp": "src/grid.cpp"})
        found = cpu.attribute(stacks, resolve)
        self.assertEqual(found["files"], {"src/solver.f90": 0.6667, "src/grid.cpp": 0.3})
        self.assertEqual(found["lines"], {"src/solver.f90:6": 0.6667, "src/grid.cpp:42": 0.3})
        self.assertEqual(found["functions"]["src/solver.f90::solver::relax"], {"self": 0.6667, "total": 0.6667, "line": 6})
        self.assertEqual(found["outside"], {"small allocations": 0.0333})

    def test_the_readers_own_format_reads_root_first(self) -> None:
        peak, stacks = memory.json_stacks({"peak": 2**20, "stacks": [
            [512, [["<module>", "main.py", 3], ["build", "pkg/store.py", 2]]]]})
        self.assertEqual(peak, 2**20)
        self.assertEqual(stacks, [([Frame("main.py", "<module>", 3), Frame("pkg/store.py", "build", 2)], 512)])

    def test_the_tool_that_fits(self) -> None:
        with mock.patch.object(memory, "_has_memray", return_value=False):
            self.assertEqual(memory.choose(["python3", "run.py"]), ("tracemalloc", ""))
            self.assertEqual(memory.choose(["python3", "-c", "x"])[0], "")
        with mock.patch.object(memory.shutil, "which", side_effect=lambda t: "/usr/bin/valgrind" if t == "valgrind" else None):
            self.assertEqual(memory.choose(["./solver"]), ("massif", ""))
        with mock.patch.object(memory.shutil, "which", return_value=None):
            self.assertIn("heaptrack", memory.choose(["./solver"])[1])

    def _python_harness(self, native: tuple[str, str], heaps: dict) -> dict:
        """A Python harness whose code allocates in a child: its own heap is small."""
        resolve = lambda p: p if p.startswith("src/") else None
        taken = lambda tool, *_a: heaps[tool]
        with mock.patch.object(memory, "choose", return_value=("memray", "")), \
             mock.patch.object(memory, "choose_native", return_value=native), \
             mock.patch.object(memory, "_take", side_effect=taken):
            return memory.profile(["python3", "h.py"], self.root, resolve, None, None, measured_mb=80)

    def test_a_python_harness_s_child_is_traced_by_the_native_profiler(self) -> None:
        heaps = {"memray": (2**20, [([Frame("bench/h.py", "<module>", 2)], 2**20)]),
                 "heaptrack": (80 * 2**20, [([Frame("src/grid.c", "build", 4)], 80 * 2**20)])}
        found = self._python_harness(("heaptrack", ""), heaps)
        self.assertEqual((found["tool"], found["files"], found["peak_mb"]), ("heaptrack", {"src/grid.c": 1.0}, 80.0))
        self.assertIn("memray traced 1 of the 80 MB measured, so heaptrack profiled", found["note"])

    def test_a_python_harness_s_child_untraced_says_why(self) -> None:
        heaps = {"memray": (2**20, [([Frame("src/h.py", "<module>", 2)], 2**20)])}
        found = self._python_harness(("", "neither heaptrack nor valgrind is installed"), heaps)
        self.assertEqual(found["tool"], "memray")
        self.assertIn("not followed: neither heaptrack nor valgrind", found["note"])

    def test_the_memory_view_is_files_then_the_lines_of_one(self) -> None:
        prof = {"tool": "py-spy", "files": {}, "memory": {
            "tool": "massif", "peak_mb": 28.6, "files": {"src/solver.f90": 0.67, "src/grid.cpp": 0.3},
            "functions": {"src/solver.f90::solver::relax": {"self": 0.67, "total": 0.67}},
            "lines": {"src/solver.f90:6": 0.67, "src/grid.cpp:42": 0.3}, "outside": {"small allocations": 0.03}}}
        view = profiling.top(prof, level="memory")
        self.assertEqual((view["peak_mb"], view["outside"]), (28.6, 0.03))
        one = profiling.top(prof, level="memory", focus="src/solver.f90")
        self.assertEqual(one["lines"], {"src/solver.f90:6": 0.67})
        self.assertEqual(list(one["functions"]), ["src/solver.f90::solver::relax"])

    def _build(self, name: str, source: str, compiler: str) -> str:
        if not shutil.which(compiler):
            self.skipTest(f"{compiler} is not installed")
        _write_files(self.root, {f"src/{name}": source})
        exe = os.path.join(self.root, "solver")
        subprocess.run([compiler, "-g", "-O1", "-o", exe, os.path.join("src", name)], cwd=self.root,
                       check=True, capture_output=True)
        return exe

    def _massif(self, exe: str, files: list[str]) -> dict:
        if not shutil.which("valgrind"):
            self.skipTest("valgrind is not installed")
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        with mock.patch.object(memory, "choose", return_value=("massif", "")):
            return profiling.profile([exe], self.root, self.root, files, level="memory", timeout=120,
                                     peak_mb=1000)["memory"]

    def test_massif_finds_the_allocation_of_a_child_a_python_harness_starts(self) -> None:
        exe = self._build("grid.c", _C_ALLOC.replace("free(x); ", ""), "cc")
        if not shutil.which("valgrind"):
            self.skipTest("valgrind is not installed")
        _write_files(self.root, {"h.py": f"import subprocess\nsubprocess.run([{exe!r}], check=True)\n"})
        with mock.patch.object(memory, "_has_memray", return_value=False), \
             mock.patch.object(memory, "choose_native", return_value=("massif", "")):
            found = profiling.profile([sys.executable, os.path.join(self.root, "h.py")], self.root, self.root,
                                      ["src/grid.c", "h.py"], level="memory", timeout=300, peak_mb=16)["memory"]
        self.assertEqual(found["tool"], "massif")
        self.assertEqual(next(iter(found["functions"])), "src/grid.c::build")
        self.assertIn("tracemalloc traced", found["note"])

    def test_massif_finds_the_fortran_allocation(self) -> None:
        exe = self._build("solver.f90", _F90_ALLOC, "gfortran")
        found = self._massif(exe, ["src/solver.f90"])
        self.assertEqual(found["tool"], "massif")
        self.assertGreater(found["peak_mb"], 15)
        self.assertEqual(next(iter(found["lines"])), "src/solver.f90:6")
        self.assertEqual(next(iter(found["functions"])), "src/solver.f90::solver::relax")
        self.assertIn("automatic arrays", found["note"])
        self.assertIn("of the 1000 MB measured", found["note"])

    def test_massif_finds_the_c_allocation(self) -> None:
        exe = self._build("grid.c", _C_ALLOC, "cc")
        found = self._massif(exe, ["src/grid.c"])
        self.assertEqual(next(iter(found["lines"])), "src/grid.c:4")
        self.assertEqual(next(iter(found["functions"])), "src/grid.c::build")


_F90_ALLOC = """\
module solver
contains
  subroutine relax(n)
    integer :: n
    real, allocatable :: x(:)
    allocate(x(n))
    x = 1.0
    print *, sum(x)
    deallocate(x)
  end subroutine
end module
program main
  use solver
  call relax(5000000)
end program
"""
_C_ALLOC = """\
#include <stdio.h>
#include <stdlib.h>
double *build(int n) {
  double *x = malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) x[i] = i;
  return x;
}
int main(void) { double *x = build(2000000); printf("%f\\n", x[7]); free(x); return 0; }
"""


class RealProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = self.tmp.name
        os.makedirs(os.path.join(root, "pkg"))
        files = {
            "pkg/__init__.py": "",
            "pkg/work.py": textwrap.dedent("""\
                import math
                def heavy(n):
                    return sum(math.sqrt(i) for i in range(n))
                """),
            "main.py": "from pkg.work import heavy\nfor _ in range(4):\n    heavy(300000)\n",
        }
        for rel, text in files.items():
            with open(os.path.join(root, rel), "w") as fh:
                fh.write(text)
        self.files = sorted(files)

    def _profile(self) -> dict:
        out = os.path.join(self.root, "out")
        os.makedirs(out, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            return profiling.profile([sys.executable, "main.py"], out, self.root, self.files, timeout=120)
        finally:
            os.chdir(cwd)

    def test_cprofile_finds_the_hot_file(self) -> None:
        real_which = shutil.which
        with mock.patch.object(cpu.shutil, "which", side_effect=lambda t: None if t == "py-spy" else real_which(t)):
            prof = self._profile()
        self.assertEqual(prof["tool"], "cprofile")
        self.assertEqual(next(iter(prof["files"])), "pkg/work.py")
        self.assertEqual(next(iter(prof["functions"])), "pkg/work.py::<genexpr>")
        # Opened where it is defined: cProfile keeps no time per line.
        self.assertGreater(prof["functions"]["pkg/work.py::<genexpr>"]["line"], 0)
        self.assertNotIn("lines", prof)
        with open(os.path.join(self.root, "out", "profile.json")) as fh:
            self.assertEqual(json.load(fh)["files"], prof["files"])

    @unittest.skipUnless(shutil.which("py-spy"), "py-spy is not installed")
    def test_py_spy_finds_the_hot_file(self) -> None:
        prof = self._profile()
        if prof.get("unavailable"):
            self.skipTest(prof["unavailable"])
        self.assertEqual(prof["tool"], "py-spy")
        self.assertGreater(prof["files"].get("pkg/work.py", 0), 0.5)
        self.assertEqual(next(iter(prof["lines"])), "pkg/work.py:3")

    @unittest.skipUnless(shutil.which("py-spy"), "py-spy is not installed")
    def test_py_spy_follows_the_python_child_a_harness_starts(self) -> None:
        _write_files(self.root, {"bench/h.py": "import subprocess, sys\n"
                                               "subprocess.run([sys.executable, 'main.py'], check=True)\n"})
        out = os.path.join(self.root, "out")
        os.makedirs(out, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            prof = profiling.profile([sys.executable, "bench/h.py"], out, self.root, self.files + ["bench/h.py"],
                                     timeout=120)
        finally:
            os.chdir(cwd)
        if prof.get("unavailable"):
            self.skipTest(prof["unavailable"])
        self.assertGreater(prof["files"].get("pkg/work.py", 0), 0.5)
        self.assertLess(prof["files"].get("bench/h.py", 0), 0.1)

    def test_tracemalloc_finds_the_line_that_allocates(self) -> None:
        _write_files(self.root, {"pkg/store.py": "def build(n):\n    return [float(i) for i in range(n)]\n",
                                 "alloc.py": "from pkg.store import build\nkeep = build(10**6)\n"})
        out = os.path.join(self.root, "out")
        os.makedirs(out, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(memory, "_has_memray", return_value=False):
                prof = profiling.profile([sys.executable, "alloc.py"], out, self.root,
                                         self.files + ["pkg/store.py", "alloc.py"], level="memory", timeout=120)
        finally:
            os.chdir(cwd)
        found = prof["memory"]
        self.assertEqual(found["tool"], "tracemalloc")
        self.assertEqual(next(iter(found["files"])), "pkg/store.py")
        self.assertEqual(next(iter(found["lines"])), "pkg/store.py:2")
        self.assertGreater(found["peak_mb"], 20)

    def test_no_profiler_is_a_profile_that_says_why(self) -> None:
        with mock.patch.object(profiling, "choose", return_value=("", "perf is blocked here")), \
             mock.patch.object(gpu, "gpu_vendor", return_value=""):
            prof = profiling.profile(["./solver"], self.root, self.root, self.files)
        self.assertEqual(prof["unavailable"], "perf is blocked here")
        self.assertEqual(profiling.top(prof), {"unavailable": "perf is blocked here"})


class RunnerProfilesTests(_TmpStorageTest):
    """The baseline and kept runs are profiled after their verdict; a ruled-out run only
    when asked; the measurement never sees it."""

    def setUp(self) -> None:
        super().setUp()
        exe = self.exe = os.path.join(self.root, "fast.py")
        with open(exe, "w") as fh:
            fh.write(textwrap.dedent("""\
                import sys
                import numpy as np
                out = sys.argv[1] if sys.argv[1].endswith('.npz') else sys.argv[1] + '.npz'
                total = sum(i * i for i in range(200000))
                np.savez(out, field=np.ones(4))
                print('PROXY_METRICS_BEGIN')
                print('time_s=0.01')
                print(f'output_file={out}')
                print('PROXY_METRICS_END')
                """))
        server_proxy.proxy_manage(op="register", name="fast", executable_path=exe,
                                  run_cmd_template=f"{sys.executable} {{executable}} {{output_file}}",
                                  output_format="npz", confirm=True)
        server_proxy.proxy_manage(op="suite_define", name="fastb",
                                  cases=[{"case_id": "a", "proxy_name": "fast"}], confirm=True)
        tracked = os.path.join(self.root, "tracked.py")
        with open(tracked, "w") as fh:
            fh.write("TUNABLE = 1\n")
        res = _eval(op="init", proxy_name="fast", benchmark_name="fastb", proxy_source_path=exe,
                    optimize_paths=[tracked], primary_metric="time_s",
                    requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}], confirm=True)
        self.assertEqual(res.get("status"), "ok")
        # The profiled command runs Python: cProfile is always there, py-spy is not.
        real_which = shutil.which
        patcher = mock.patch.object(cpu.shutil, "which",
                                    side_effect=lambda t: None if t == "py-spy" else real_which(t))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, profile: bool = False) -> str:
        cfg, error, run_dir, _notice = eval_session.prepare_run("fast", "", profile)
        self.assertIsNone(error)
        procs.update_opt_active_link("fast", run_dir)
        from mimir.servers.proxy import _proxy_runner
        with mock.patch.object(sys, "argv", ["_proxy_runner", "--run-dir", run_dir]):
            _proxy_runner.main()
        return run_dir

    def test_the_baseline_is_profiled_and_a_ruled_out_run_is_not(self) -> None:
        base = self._run()
        prof = read_json(profiling.profile_path(base))
        self.assertEqual(prof["tool"], "cprofile")
        self.assertIn("fast.py", prof["files"])
        # The same time again is no improvement: rejected, so not profiled.
        second = self._run()
        self.assertIsNone(read_json(profiling.profile_path(second)))
        third = self._run(profile=True)
        self.assertIsNotNone(read_json(profiling.profile_path(third)))

    def test_results_say_where_the_time_goes(self) -> None:
        self._run()
        res = server_proxy.proxy_eval_status(op="results", proxy_name="fast")
        self.assertEqual(res["verdict"], "accept")
        self.assertIn("fast.py", res["where_time_goes"]["files"])

    def test_profiling_the_current_tree_files_it_with_the_run_that_measured_it(self) -> None:
        base = self._run()
        os.remove(profiling.profile_path(base))
        from mimir.servers.proxy import _proxy_runner

        real_run = eval_session.proc_run.run

        def in_process(argv, **kwargs):
            # The runner in this process, where the test's storage is; anything else runs.
            if "--run-dir" not in argv:
                return real_run(argv, **kwargs)
            with mock.patch.object(sys, "argv", ["_proxy_runner", *argv[argv.index("--run-dir"):]]):
                _proxy_runner.main()
        with mock.patch.object(eval_session.proc_run, "run", side_effect=in_process):
            res = _eval(op="profile", proxy_name="fast", confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["profiled"], os.path.basename(base))
        self.assertIn("fast.py", res["where_time_goes"]["files"])
        self.assertIsNotNone(read_json(profiling.profile_path(base)))

    def test_a_deeper_view_of_a_profiled_tree_runs_nothing(self) -> None:
        base = self._run()
        with mock.patch.object(eval_session.proc_run, "run", side_effect=AssertionError("ran")):
            res = _eval(op="profile", proxy_name="fast", level="function", focus="fast.py", confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["from_profile_of"], os.path.basename(base))
        self.assertTrue(res["where_time_goes"]["functions"])
        self.assertTrue(all(k.startswith("fast.py::") for k in res["where_time_goes"]["functions"]))

    def test_a_memory_profile_joins_the_run_s_and_is_read_again_without_a_run(self) -> None:
        base = self._run()
        from mimir.servers.proxy import _proxy_runner

        def in_process(argv, **kwargs):
            with mock.patch.object(sys, "argv", ["_proxy_runner", *argv[argv.index("--run-dir"):]]):
                _proxy_runner.main()
        found = {"tool": "massif", "peak_mb": 12.0, "files": {"fast.py": 1.0},
                 "functions": {}, "lines": {"fast.py:4": 1.0}, "outside": {}}
        with mock.patch.object(eval_session.proc_run, "run", side_effect=in_process), \
             mock.patch.object(memory, "profile", return_value=found):
            res = _eval(op="profile", proxy_name="fast", level="memory", confirm=True)
        self.assertEqual(res["where_time_goes"]["files"], {"fast.py": 1.0})
        self.assertEqual(read_json(profiling.profile_path(base))["memory"], found)
        with mock.patch.object(eval_session.proc_run, "run", side_effect=AssertionError("ran")):
            again = _eval(op="profile", proxy_name="fast", level="memory", focus="fast.py", confirm=True)
        self.assertEqual(again["where_time_goes"]["lines"], {"fast.py:4": 1.0})

    def test_a_list_of_levels_gives_one_view_each_and_keeps_every_section(self) -> None:
        base = self._run()
        from mimir.servers.proxy import _proxy_runner
        runs = []
        real_run = eval_session.proc_run.run

        def in_process(argv, **kwargs):
            if "--run-dir" not in argv:
                return real_run(argv, **kwargs)
            run_dir = argv[argv.index("--run-dir") + 1]
            runs.append(read_json(os.path.join(run_dir, "config.json"))["profile_level"])
            with mock.patch.object(sys, "argv", ["_proxy_runner", *argv[argv.index("--run-dir"):]]):
                _proxy_runner.main()
        found = {"tool": "massif", "peak_mb": 12.0, "files": {"fast.py": 1.0}, "functions": {}, "lines": {}, "outside": {}}
        with mock.patch.object(eval_session.proc_run, "run", side_effect=in_process), \
             mock.patch.object(memory, "profile", return_value=found), \
             mock.patch.object(cpu, "counters", return_value={"tool": "perf stat", "ipc": 0.7}):
            res = _eval(op="profile", proxy_name="fast", level=["memory", "function", "counters"], confirm=True)
            self.assertEqual(res["status"], "ok")
            # The function view is read from the run's profile; the others take a run each.
            self.assertEqual(runs, ["counters", "memory"])
            views = res["where_time_goes"]
            self.assertEqual(list(views), ["function", "counters", "memory"])
            self.assertEqual(views["memory"]["peak_mb"], 12.0)
            self.assertEqual(views["counters"]["ipc"], 0.7)
            # cProfile keeps no lines: a line view takes a sampled run, which keeps the sections.
            res = _eval(op="profile", proxy_name="fast", level=["line"], confirm=True)
        self.assertIn("unavailable", res["where_time_goes"])
        kept = read_json(profiling.profile_path(base))
        self.assertEqual((kept["memory"], kept["counters"]["ipc"]), (found, 0.7))
        self.assertIn("fast.py", kept["files"])

    def test_an_unknown_level_in_the_list_is_named(self) -> None:
        self._run()
        res = _eval(op="profile", proxy_name="fast", level=["file", "heap"], confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("heap", res["error"])

    def test_counters_join_the_profile_the_run_has(self) -> None:
        base = self._run()
        from mimir.servers.proxy import _proxy_runner

        def in_process(argv, **kwargs):
            with mock.patch.object(sys, "argv", ["_proxy_runner", *argv[argv.index("--run-dir"):]]):
                _proxy_runner.main()
        with mock.patch.object(eval_session.proc_run, "run", side_effect=in_process), \
             mock.patch.object(cpu, "counters", return_value={"tool": "perf stat", "ipc": 0.7}):
            res = _eval(op="profile", proxy_name="fast", level="counters", confirm=True)
        self.assertEqual(res["where_time_goes"], {"tool": "perf stat", "ipc": 0.7})
        kept = read_json(profiling.profile_path(base))
        self.assertEqual(kept["counters"], {"tool": "perf stat", "ipc": 0.7})
        self.assertIn("fast.py", kept["files"])


if __name__ == "__main__":
    unittest.main()
