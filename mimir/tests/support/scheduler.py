"""Pin whether this machine has its own Slurm, for the rest of one test.

Scheduler resolution starts from a probe of the local machine, made once at import.
A test that does not pin it depends on the host it runs on.
"""
from __future__ import annotations

import unittest

from mimir.shared import slurm_site


def local_slurm(case: unittest.TestCase, present: bool = True) -> None:
    """Answer the local-Slurm probe with *present* until *case* ends."""
    original = slurm_site._HAS_SLURM
    slurm_site._HAS_SLURM = present
    case.addCleanup(setattr, slurm_site, "_HAS_SLURM", original)


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
