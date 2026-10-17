"""Pointing the client's state directory somewhere disposable."""
from __future__ import annotations

import unittest
from unittest import mock

from mimir.client.config import paths


def pin_state_dir(case: unittest.TestCase, directory: str) -> None:
    """Make *directory* the state dir for the rest of *case*."""
    patcher = mock.patch.object(paths, "STATE_DIR", directory)
    patcher.start()
    case.addCleanup(patcher.stop)
