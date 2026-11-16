"""Pin which scheduler this machine has of its own, for the rest of one test.

Scheduler resolution starts from a probe of the local machine, made once at import.
A test that does not pin it depends on the host it runs on.
"""
from __future__ import annotations

import unittest

from mimir.shared import cluster_site


def local_slurm(case: unittest.TestCase, present: bool = True) -> None:
    """Answer the local-scheduler probe with Slurm, or none, until *case* ends."""
    local_scheduler(case, "slurm" if present else "")


def local_scheduler(case: unittest.TestCase, kind: str) -> None:
    """Answer the local-scheduler probe with *kind* (``slurm``, ``lsf`` or "") until *case* ends."""
    original = cluster_site._LOCAL_KIND
    cluster_site._LOCAL_KIND = kind
    case.addCleanup(setattr, cluster_site, "_LOCAL_KIND", original)


def reads_through_stubs(case: unittest.TestCase) -> None:
    """Send ``cli.capture``, the seam the Slurm reads go through, to whatever stands for
    ``cli.run_argv`` (a scontrol read) or ``cli.run_bash`` (the rest) when it is called,
    so a test's canned answers serve both, until *case* ends."""
    import shlex

    from mimir.servers.hpc import cli

    def capture(script: str, timeout: int, *, host: str = "") -> str | None:
        argv = shlex.split(script) if script.startswith("scontrol show node") else None
        res = cli.run_argv(argv, timeout, host=host) if argv else cli.run_bash(script, timeout, host=host)
        return res.get("stdout", "") if res.get("status") == "ok" else None

    original = cli.capture
    cli.capture = capture
    case.addCleanup(setattr, cli, "capture", original)
