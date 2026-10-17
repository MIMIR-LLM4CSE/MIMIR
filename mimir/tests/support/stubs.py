"""Replacing an attribute for the rest of one test, and putting it back after."""
from __future__ import annotations

import unittest
from typing import Any
from unittest import mock


def stub_attr(case: unittest.TestCase, obj: Any, name: str, value: Any) -> None:
    patcher = mock.patch.object(obj, name, value)
    patcher.start()
    case.addCleanup(patcher.stop)
