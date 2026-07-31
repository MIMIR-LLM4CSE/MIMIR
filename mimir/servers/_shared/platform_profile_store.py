"""Shared persistence helpers for the MCP platform profile."""

import json
import logging
import os
from datetime import datetime, timezone

from state_paths import state_dir

_log = logging.getLogger(__name__)

# Central per-workspace state dir (MIMIR_STATE_DIR, set by server_manager), with a
# legacy <workspace>/.mimir fallback for standalone/test runs. See state_paths.
PROFILE_FILE = os.path.join(state_dir(), "platform_profile.json")



def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def load_profile() -> dict:
    if not os.path.exists(PROFILE_FILE):
        return {}
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _log.warning("Failed to load profile %s: %s", PROFILE_FILE, exc)
        return {}



def save_profile(profile: dict) -> bool:
    try:
        os.makedirs(os.path.dirname(PROFILE_FILE), exist_ok=True)
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        _log.warning("Failed to save profile %s: %s", PROFILE_FILE, exc)
        return False



def empty_profile_base() -> dict:
    return {
        "timestamp": now_iso(),
        "hostname": "",
        "platform": "",
        "cpu": {},
        "memory": {},
        "gpu": {},
        "slurm": {},
        "modules": {},
        "toolchains": {},
    }



_REQUIRED_REPORT_KEYS = {"timestamp"}


def persist_benchmark_report(report: dict) -> dict:
    missing = _REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        report = dict(report)
        report.setdefault("timestamp", now_iso())

    profile = load_profile() or empty_profile_base()

    profile.setdefault("benchmarks", {})
    profile["benchmarks"]["latest"] = report
    profile["benchmarks"].setdefault("history", [])
    profile["benchmarks"]["history"].append(
        {
            "timestamp": report.get("timestamp", now_iso()),
            "summary": {
                "python_mops": report.get("python_compute", {}).get("throughput_mops"),
                "memory_gbps": report.get("memory_copy", {}).get("throughput_gbps"),
                "numpy_gflops": report.get("numpy_matmul", {}).get("gflops"),
            },
        }
    )
    profile["benchmarks"]["history"] = profile["benchmarks"]["history"][-20:]
    profile["timestamp"] = now_iso()

    persisted = save_profile(profile)
    return {
        "persisted": persisted,
        "profile_path": PROFILE_FILE,
        "history_size": len(profile["benchmarks"]["history"]),
    }
