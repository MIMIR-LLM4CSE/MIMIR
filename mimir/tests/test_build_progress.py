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

    def test_a_job_without_a_count_carries_neither_field(self) -> None:
        result = server_bash.bash_run("echo hello; sleep 2", background=True)
        time.sleep(0.3)
        payload = _bash_jobs.state(result["job_key"])
        self.assertEqual(payload["state"], "running")
        self.assertNotIn("percent", payload)
        self.assertNotIn("phase", payload)


if __name__ == "__main__":
    unittest.main()
