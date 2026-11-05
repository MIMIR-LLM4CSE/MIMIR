"""What a build leaves in a run directory: its log and its report. Read by the server's
status and the runner as much as written by the build, so neither side owns them."""
from __future__ import annotations

import os

from ....shared.atomic import read_json


def build_log_path(run_dir: str) -> str:
    return os.path.join(run_dir, "build.log")


def report_path(run_dir: str) -> str:
    return os.path.join(run_dir, "build.json")


def read_report(run_dir: str) -> dict | None:
    """Read a run's build.json, or None when the run declared no build."""
    data = read_json(report_path(run_dir))
    return data if isinstance(data, dict) else None
