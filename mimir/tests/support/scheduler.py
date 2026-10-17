"""Pin whether this machine has its own Slurm, for the rest of one test.

Scheduler resolution starts from a probe of the local machine, made once at import.
A test that does not pin it depends on the host it runs on.
"""
from __future__ import annotations

import unittest

from mimir.servers._shared import slurm_site


def local_slurm(case: unittest.TestCase, present: bool = True) -> None:
    """Answer the local-Slurm probe with *present* until *case* ends."""
    original = slurm_site._HAS_SLURM
    slurm_site._HAS_SLURM = present
    case.addCleanup(setattr, slurm_site, "_HAS_SLURM", original)
