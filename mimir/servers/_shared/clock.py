"""The current time in UTC, in the three forms the servers write."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_stamp() -> str:
    """``2026-10-17T08:10:46Z``: what a record stores."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_minute() -> str:
    """``2026-10-17 08:10``: what a person reads."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def utc_date() -> str:
    """``2026-10-17``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
