"""A build's progress, read from its own output, for any shell command.

The parser first (``_shared/build_progress``), then the two places that carry it to
the UI: a blocking ``bash_run`` republishing it on the run channel, and a detached
job's status carrying it for the client watcher. Neither place knows the command was
a build — that is the point: a ``make`` buried in a chain still prints its count.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_progress  # noqa: E402
import run_channel  # noqa: E402
import _bash_jobs  # noqa: E402
import server_bash  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_cmake_and_fpm_percentages(self) -> None:
        self.assertEqual(
            build_progress.parse("[  6%] Building A\n[ 42%] Building CXX object b.o\n"),
            (42.0, "[ 42%] Building CXX object b.o"))

    def test_ninja_ratio(self) -> None:
        self.assertEqual(build_progress.parse("[12/48] Building CXX object a.o\n")[0], 25.0)

    def test_bazel_ratio_with_thousands(self) -> None:
        self.assertEqual(
            build_progress.parse("[1,000 / 4,000] Compiling x.cc\n")[0], 25.0)

    def test_the_newest_count_wins_over_the_largest(self) -> None:
        # Several sub-builds restart the count; the run is where it last said it was.
        self.assertEqual(build_progress.parse("[ 90%] a\n[ 10%] b\n")[0], 10.0)

    def test_warnings_after_a_count_keep_it(self) -> None:
        text = "[ 37%] Building a.o\nwarning: unused variable\n   int x;\n"
        self.assertEqual(build_progress.parse(text)[0], 37.0)

    def test_a_finished_build_followed_by_other_work_reports_nothing(self) -> None:
        # make && ./bench: a full bar standing over the benchmark would be a lie.
        self.assertIsNone(build_progress.parse("[100%] Built target bench\nstep 1\n"))
        self.assertEqual(build_progress.parse("[100%] Built target bench\n")[0], 100.0)

    def test_a_finished_build_is_kept_when_the_caller_knows_it_is_current(self) -> None:
        self.assertEqual(
            build_progress.parse("[100%] Linking\nld: note\n", drop_finished=False)[0],
            100.0)

    def test_output_without_a_count_reports_nothing(self) -> None:
        for text in ("", "gcc -c a.c\n",
                     "(./chapter1.tex [1] [2])\n",           # pdflatex pages, no total
                     "error: expected [3/4] here\n",          # not at line start
                     "[7/0] nonsense\n", "[9/4] nonsense\n"):
            with self.subTest(text=text):
                self.assertIsNone(build_progress.parse(text))

    def test_a_redrawn_status_line_reads_its_last_state(self) -> None:
        self.assertEqual(build_progress.parse("[1/10] a\r[2/10] b\r[3/10] c")[0], 30.0)

    def test_a_long_line_is_clipped(self) -> None:
        percent, phase = build_progress.parse("[ 5%] " + "x" * 500 + "\n")
        self.assertEqual(percent, 5.0)
        self.assertLessEqual(len(phase), 160)

    def test_a_tail_does_not_read_the_line_it_cut(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            fh.write("x[3/4] a\nz\n")
        try:
            # The whole file has no count at line start ("x[3/4]"). A 10-byte tail
            # starts right after the "x", where the fragment would read as one.
            self.assertIsNone(build_progress.from_file(fh.name))
            self.assertEqual(build_progress.read_tail(fh.name, 10), "z\n")
            self.assertIsNone(build_progress.from_file(fh.name + ".missing"))
        finally:
            os.unlink(fh.name)


class TeeInjectionTests(unittest.TestCase):
    """Which commands get a second copy of their output, and where it is spliced.

    The log a job keeps is the *pipeline's* output, so a filter at the end of one
    holds back exactly what the bar is read from. These decide when a `tee` goes in
    ahead of the filter — the one judgement in this module made from the command
    line rather than from the output, because it has to be made before there is any.
    """
    LOG = "/jobs/k/progress.log"

    def _teed(self, command: str) -> str | None:
        return build_progress.tee_command(command, self.LOG)

    def test_a_piped_build_gets_the_tee_before_the_filter(self) -> None:
        self.assertEqual(
            self._teed("make -j32 2>&1 | tail -40"),
            "make -j32 2>&1 | tee -a /jobs/k/progress.log | tail -40")

    def test_the_build_may_sit_anywhere_in_the_chain(self) -> None:
        for command, expected in (
            ("cd build && make -j8 2>&1 | tail -n 40",
             "cd build && make -j8 2>&1 | tee -a /jobs/k/progress.log | tail -n 40"),
            ("timeout 600 make 2>&1 | tail -5",
             "timeout 600 make 2>&1 | tee -a /jobs/k/progress.log | tail -5"),
            ("meson compile -C build | tail",
             "meson compile -C build | tee -a /jobs/k/progress.log | tail"),
            # Not only `tail`: any filter withholds the output just as well.
            ("ninja | grep -i error",
             "ninja | tee -a /jobs/k/progress.log | grep -i error"),
        ):
            with self.subTest(command=command):
                self.assertEqual(self._teed(command), expected)

    def test_nothing_is_touched_when_the_log_already_sees_the_build(self) -> None:
        for command in (
            "make -j8 2>&1",            # unpiped: run.log already has every line
            "make && tail run.log",     # '&&' is not a pipe — the memory's false case
            "make 2>&1 | tee mine.log | tail -5",   # already copied somewhere
            "cmake -S . -B build | tail -5",        # configure, and it prints no count
            "pip list | grep numpy",                # not a build
            "grep -rn 'a | b' src | tail -5",       # the pipe is inside a quote
        ):
            with self.subTest(command=command):
                self.assertIsNone(self._teed(command))

    def test_a_refused_command_is_left_exactly_as_it_came(self) -> None:
        self.assertIsNone(self._teed("make $(whoami) | tail -3"))

    def test_the_quoted_pipe_does_not_shift_the_splice(self) -> None:
        # The parse counts one pipe separator; a naive scan would find two and cut
        # inside the pattern, producing a command that no longer greps what it said.
        self.assertEqual(
            self._teed("make -j8 | grep -E 'a|b'"),
            "make -j8 | tee -a /jobs/k/progress.log | grep -E 'a|b'")


class _JobsSandbox(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="mimir-bash-jobs-")
        self._orig_root = _bash_jobs.JOBS_ROOT
        _bash_jobs.JOBS_ROOT = self._tmp
        self._state = tempfile.mkdtemp(prefix="mimir-bash-state-")
        self._orig_state = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._state

    def tearDown(self) -> None:
        for payload in _bash_jobs.listing():
            if payload["state"] == "running":
                _bash_jobs.stop(payload["job_key"])
        _bash_jobs.JOBS_ROOT = self._orig_root
        if self._orig_state is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig_state
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._state, ignore_errors=True)


class ForegroundTests(_JobsSandbox):
    def test_a_build_count_inside_a_chain_reaches_the_run_channel(self) -> None:
        seen: list[tuple] = []
        path = os.path.join(run_channel._dir("bash_run"), "current.json")

        def watch() -> None:
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    with open(path) as fh:
                        run = json.load(fh)
                    pair = (run.get("phase"), run.get("percent"))
                    if not seen or seen[-1] != pair:
                        seen.append(pair)
                except (OSError, ValueError):
                    pass
                time.sleep(0.05)

        threading.Thread(target=watch, daemon=True).start()
        result = server_bash.bash_run(
            "export X=1 && echo '[ 42%] Building CXX object a.o' && sleep 2.5"
            " && echo '[100%] Built target a' && echo running && sleep 2.5",
            timeout=30)
        self.assertEqual(result["returncode"], 0)
        percents = [p for _, p in seen]
        self.assertIn(42.0, percents)
        self.assertIn(("[ 42%] Building CXX object a.o", 42.0), seen)
        # Once the build is over and the chain moved on, the bar is retracted.
        after = percents[percents.index(42.0):]
        self.assertIn(None, after)

    def test_a_command_without_a_count_publishes_none(self) -> None:
        path = os.path.join(run_channel._dir("bash_run"), "current.json")
        percents: list = []

        def watch() -> None:
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    with open(path) as fh:
                        percents.append(json.load(fh).get("percent"))
                except (OSError, ValueError):
                    pass
                time.sleep(0.05)

        threading.Thread(target=watch, daemon=True).start()
        server_bash.bash_run("echo compiling quietly; sleep 1.5", timeout=30)
        self.assertTrue(percents)
        self.assertEqual(set(percents), {None})


class BackgroundTests(_JobsSandbox):
    def test_a_running_job_status_carries_the_count(self) -> None:
        result = server_bash.bash_run(
            "cd . && echo '[3/12] Building a.o' && sleep 3", background=True)
        job_key = result["job_key"]
        deadline = time.time() + 5
        payload = _bash_jobs.state(job_key)
        while "percent" not in payload and time.time() < deadline:
            time.sleep(0.05)
            payload = _bash_jobs.state(job_key)
        self.assertEqual(payload["state"], "running")
        self.assertEqual(payload["percent"], 25.0)
        self.assertEqual(payload["phase"], "[3/12] Building a.o")

    def test_a_build_piped_into_tail_still_reports_its_count(self) -> None:
        """The whole point: a bar during the run, the filter's output after it.

        A stand-in `make` on PATH, since the count has to arrive line by line over
        several seconds for there to be anything to observe while the job runs.
        """
        bin_dir = os.path.join(self._tmp, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        fake = os.path.join(bin_dir, "make")
        with open(fake, "w") as fh:
            fh.write("#!/bin/bash\nfor i in 10 40 70 100; do\n"
                     "  echo \"[ $i%] Building CXX object x$i.o\"\n  sleep 0.6\ndone\n")
        os.chmod(fake, 0o755)
        env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])

        job = _bash_jobs.launch("make -j8 2>&1 | tail -2", cwd=self._tmp, env=env)
        job_key = job["job_key"]
        try:
            seen: list[float] = []
            deadline = time.time() + 10
            while job["proc"].poll() is None and time.time() < deadline:
                found = build_progress.from_file(_bash_jobs.progress_path(job_key))
                if found and (not seen or seen[-1] != found[0]):
                    seen.append(found[0])
                    # While `tail` holds the pipeline's output, the job's own log is
                    # empty — the copy is the only place the count can be read.
                    if found[0] < 100:
                        self.assertEqual(open(_bash_jobs.log_path(job_key)).read(), "")
                time.sleep(0.1)
            self.assertEqual(job["proc"].wait(), 0)
            self.assertIn(10.0, seen)
            self.assertIn(70.0, seen)
            # The caller still gets what it asked for, filtered and nothing more.
            self.assertEqual(
                open(_bash_jobs.log_path(job_key)).read(),
                "[ 70%] Building CXX object x70.o\n[ 100%] Building CXX object x100.o\n")
        finally:
            _bash_jobs.discard(job_key)

    def test_a_job_without_a_count_carries_neither_field(self) -> None:
        result = server_bash.bash_run("echo hello; sleep 2", background=True)
        time.sleep(0.3)
        payload = _bash_jobs.state(result["job_key"])
        self.assertEqual(payload["state"], "running")
        self.assertNotIn("percent", payload)
        self.assertNotIn("phase", payload)


if __name__ == "__main__":
    unittest.main()
