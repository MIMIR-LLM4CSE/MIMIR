"""Shared fixtures for the proxy test modules.

Underscore-prefixed so ``unittest discover`` does not collect it: it holds no tests.

It exists because ``test_proxy_ops`` had become the fixture module for three of its
siblings — re-exporting eleven server modules alongside its own thirty-seven tests, so
those siblings could not run without it and a change to either role touched the other.
A fixture module that is also a test module is two things sharing one name.

What is here is what was otherwise rewritten: the storage redirection (twelve copies of
it), a registered proxy, an initialised session, and a fake ``sbatch``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "proxy"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import server_proxy  # noqa: E402
from _lib import build, procs, ratchet, store, tree_snapshot  # noqa: E402,F401
from _ops import (  # noqa: E402,F401
    _eval_ratchet, _eval_validate, eval_session, references, registry, runs,
    scaffold, slurm, suites,
)


def _eval(*args, **kwargs):
    """Call the async ``proxy_eval`` from a synchronous test.

    ``op='run'`` awaits the run it launches, so the tool is a coroutine function.
    """
    return asyncio.run(server_proxy.proxy_eval(*args, **kwargs))


def _call(tool, **kwargs):
    """Call a proxy tool whether or not it is a coroutine function.

    The dispatch tests sweep every tool with the same arguments; which ones are
    async is not what they are testing.
    """
    res = tool(**kwargs)
    return asyncio.run(res) if inspect.isawaitable(res) else res


def _script(path: str, body: str) -> str:
    """Write an executable ``/bin/sh`` script at *path* and return it."""
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n" + body)
    os.chmod(path, 0o755)
    return path


class _TmpStorageTest(unittest.TestCase):
    """Base: redirects the proxy storage root into a per-test temp dir.

    Every path is derived from ``store._CACHE_DIR`` at call time, so one repoint
    makes the entire store hermetic.

    The temp dir doubles as the WORKSPACE (``MCP_FILES_ROOT``): ``optimize_paths``
    are refused outside it, so a fixture with no workspace of its own would be
    measuring the repository it runs from.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self._tmp.name
        self.root = self._tmp.name
        self._saved_ws = os.environ.get("MCP_FILES_ROOT")
        os.environ["MCP_FILES_ROOT"] = self._tmp.name

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        if self._saved_ws is None:
            os.environ.pop("MCP_FILES_ROOT", None)
        else:
            os.environ["MCP_FILES_ROOT"] = self._saved_ws
        self._tmp.cleanup()

    # -- fixtures ------------------------------------------------------------

    def _tracked(self) -> str:
        """A file the ratchet may edit, distinct from the harness.

        init refuses proxy_source_path being one of optimize_paths: a harness that
        is its own subject means optimizing a copy, and the accuracy constraints
        then say nothing about the code that ships.
        """
        path = os.path.join(self.root, "tracked.py")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write("TUNABLE = 1\n")
        return path

    def _make_exe(self, name: str = "tiny.py") -> str:
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            fh.write("print('PROXY_METRICS_BEGIN')\n"
                     "print('time_s=0.1')\n"
                     "print('PROXY_METRICS_END')\n")
        return path

    def _register(self, name: str = "tiny", metadata: dict | None = None) -> dict:
        return server_proxy.proxy_manage(
            op="register", name=name, executable_path=self._make_exe(),
            run_cmd_template="python3 {executable}", metadata=metadata, confirm=True,
        )

    def _session(self, metadata: dict | None = None, **init_kwargs) -> dict:
        """A registered proxy, a one-case suite, and an initialised eval session."""
        self._register(metadata=metadata)
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        src = self._make_exe("source.py")
        kwargs = {
            "requirements": [{"metric": "time_s", "operator": "lt", "threshold": 2.0}],
            "max_hours": 3.0,
        }
        kwargs.update(init_kwargs)
        return _eval(op="init", proxy_name="tiny", benchmark_name="bench",
                     proxy_source_path=src, optimize_paths=[self._tracked()],
                     confirm=True, **kwargs)


class _FakeSbatchTest(_TmpStorageTest):
    """A ``sbatch`` on PATH that hands out job ids and keeps the scripts it was given.

    Ids increment so a chained submission can be told apart job by job, and each
    script's path is recorded so its contents can be asserted on. PATH is *replaced*
    rather than prepended, so no real scheduler can be reached by accident — which
    is also why the fake uses shell builtins only: not even ``cat`` is on that PATH.
    """

    def setUp(self) -> None:
        super().setUp()
        self.bin_dir = os.path.join(self.root, "bin")
        os.makedirs(self.bin_dir, exist_ok=True)
        self._saved_path = os.environ["PATH"]
        self.counter = os.path.join(self.root, "sbatch_counter")
        self.calls_file = os.path.join(self.root, "sbatch_calls")
        _script(os.path.join(self.bin_dir, "sbatch"),
                "n=7000\n"
                f"if [ -f {self.counter} ]; then read n < {self.counter}; fi\n"
                "n=$((n+1))\n"
                f"echo $n > {self.counter}\n"
                f'echo "$1" >> {self.calls_file}\n'
                'echo "Submitted batch job $n"\n')
        os.environ["PATH"] = self.bin_dir

    def tearDown(self) -> None:
        os.environ["PATH"] = self._saved_path
        super().tearDown()

    def _submitted_scripts(self) -> list[str]:
        """The contents of every script handed to sbatch, in submission order."""
        if not os.path.isfile(self.calls_file):
            return []
        with open(self.calls_file) as fh:
            paths = [line.strip() for line in fh if line.strip()]
        return [open(p).read() for p in paths]
