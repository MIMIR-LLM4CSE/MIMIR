"""Where a run's time goes, by workspace file: the profilers' outputs read back, each
sample given to the deepest workspace frame of its stack, the profiler that fits chosen
(or why none does), and the runner profiling after the verdict without touching it."""

import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from mimir.servers.proxy._lib import layout, procs, profiling
from mimir.servers.proxy import server_proxy
from mimir.servers.proxy._ops import eval_session
from mimir.tests.support.proxy import _TmpStorageTest, _eval

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
        self.assertEqual(stacks[0], (["libc.so.6", "main.py", "pkg/work.py", "libm.so.6"], 3))

    def test_a_sample_goes_to_its_deepest_workspace_frame(self) -> None:
        resolve = _resolve({"main.py": "main.py", "pkg/work.py": "pkg/work.py"})
        prof = profiling.attribute(profiling.pyspy_stacks(_PYSPY), resolve)
        # libm called from heavy() is heavy's time; numpy's import is outside.
        self.assertEqual(prof["samples"], 10)
        self.assertEqual(prof["files"], {"pkg/work.py": 0.9})
        self.assertEqual(prof["outside"], {"numpy": 0.1})

    def test_perf_script_frames_use_the_source_file_when_perf_knows_it(self) -> None:
        stacks = profiling.perf_stacks(_PERF)
        self.assertEqual(stacks[0][0], ["/ws/src/main.c", "/ws/src/kernel.f90"])
        self.assertEqual(stacks[1][0], ["/ws/src/main.c", "/usr/lib/libc.so.6"])
        resolve = _resolve({"/ws/src/main.c": "src/main.c", "/ws/src/kernel.f90": "src/kernel.f90"})
        prof = profiling.attribute(stacks, resolve)
        self.assertEqual(prof["files"], {"src/kernel.f90": 0.3333, "src/main.c": 0.3333})
        self.assertEqual(prof["outside"], {"openblas": 0.3333})

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
        with mock.patch.object(profiling.shutil, "which", return_value="/bin/py-spy"):
            self.assertEqual(profiling.choose(["python3", "run.py"]), ("py-spy", ""))
        with mock.patch.object(profiling.shutil, "which", return_value=None):
            self.assertEqual(profiling.choose(["python3", "run.py"]), ("cprofile", ""))
            tool, why = profiling.choose(["python3", "-c", "x"])
            self.assertEqual(tool, "")
            self.assertIn("cProfile", why)

    def test_a_compiled_run_needs_perf_and_says_how_to_allow_it(self) -> None:
        with mock.patch.object(profiling, "perf_allowed", return_value=(True, "")):
            self.assertEqual(profiling.choose(["./solver"]), ("perf", ""))
        with mock.patch.object(profiling.shutil, "which", return_value="/usr/bin/perf"), \
             mock.patch.object(profiling.os, "geteuid", return_value=1000), \
             mock.patch("builtins.open", mock.mock_open(read_data="4\n")):
            tool, why = profiling.choose(["./solver"])
        self.assertEqual(tool, "")
        self.assertIn("perf_event_paranoid = 4", why)
        self.assertIn("sysctl", why)


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
        with mock.patch.object(profiling.shutil, "which", side_effect=lambda t: None if t == "py-spy" else real_which(t)):
            prof = self._profile()
        self.assertEqual(prof["tool"], "cprofile")
        self.assertEqual(next(iter(prof["files"])), "pkg/work.py")
        with open(os.path.join(self.root, "out", "profile.json")) as fh:
            self.assertEqual(json.load(fh)["files"], prof["files"])

    @unittest.skipUnless(shutil.which("py-spy"), "py-spy is not installed")
    def test_py_spy_finds_the_hot_file(self) -> None:
        prof = self._profile()
        if prof.get("unavailable"):
            self.skipTest(prof["unavailable"])
        self.assertEqual(prof["tool"], "py-spy")
        self.assertGreater(prof["files"].get("pkg/work.py", 0), 0.5)

    def test_no_profiler_is_a_profile_that_says_why(self) -> None:
        with mock.patch.object(profiling, "choose", return_value=("", "perf is blocked here")):
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
        patcher = mock.patch.object(profiling.shutil, "which",
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
        prof = layout.read_json(profiling.profile_path(base))
        self.assertEqual(prof["tool"], "cprofile")
        self.assertIn("fast.py", prof["files"])
        # The same time again is no improvement: rejected, so not profiled.
        second = self._run()
        self.assertIsNone(layout.read_json(profiling.profile_path(second)))
        third = self._run(profile=True)
        self.assertIsNotNone(layout.read_json(profiling.profile_path(third)))

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
        self.assertIsNotNone(layout.read_json(profiling.profile_path(base)))


if __name__ == "__main__":
    unittest.main()
