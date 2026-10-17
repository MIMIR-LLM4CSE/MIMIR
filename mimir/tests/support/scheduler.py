"""Pin whether this machine has its own Slurm, for the rest of one test.

Scheduler resolution starts from a probe of the local machine, made once at import.
A test that does not pin it depends on the host it runs on.
"""
from __future__ import annotations

import os
import sys
import unittest

_SHARED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "servers", "_shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import slurm_site  # noqa: E402


def local_slurm(case: unittest.TestCase, present: bool = True) -> None:
    """Answer the local-Slurm probe with *present* until *case* ends."""
    original = slurm_site._HAS_SLURM
    slurm_site._HAS_SLURM = present
    case.addCleanup(setattr, slurm_site, "_HAS_SLURM", original)
