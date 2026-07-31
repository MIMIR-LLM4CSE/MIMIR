"""Self-contained client-side hardware probe.

Produces the foundational platform profile injected as agent-identity context,
without depending on the HPC platform server being registered. Deliberately minimal
and cheap: a few best-effort subprocesses (lscpu / free / nvidia-smi) and toolchain
version lookups — no benchmarks. The platform/benchmark servers remain available for
the model's deeper, on-demand hardware queries and performance measurements.

The output dict matches the schema ``summarize_platform_profile`` renders, so the
same summarizer serves both this probe and the richer server-written profile.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
from typing import Any


def _run(cmd: list[str], timeout: int = 5) -> str:
    """Best-effort command capture; empty string on any failure."""
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=timeout,
        )
        return res.stdout if res.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu() -> dict[str, Any]:
    data: dict[str, Any] = {"arch": platform.machine(), "logical_cpus": os.cpu_count()}
    if shutil.which("lscpu"):
        info: dict[str, str] = {}
        for line in _run(["lscpu"]).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip().lower()] = v.strip()
        if info:
            flags = info.get("flags", "").split()
            data["model"] = info.get("model name", "")
            data["numa_nodes"] = info.get("numa node(s)", "")
            data["simd"] = {
                "avx2": "avx2" in flags,
                "avx512f": "avx512f" in flags,
                "fma": "fma" in flags,
            }
    return data


def _memory() -> dict[str, Any]:
    # /proc/meminfo is cheaper and more portable than shelling out to `free`.
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return {"total_gb": round(kb / 1e6, 2)}
    except (OSError, ValueError, IndexError):
        pass
    return {}


def _gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return {"available": bool(names), "count": len(names)} if names else {"available": False}


def _toolchains() -> dict[str, str]:
    out: dict[str, str] = {}
    for tool in ("gcc", "g++", "nvcc", "python3"):
        if not shutil.which(tool):
            continue
        flag = "--version"
        text = _run([tool, flag])
        m = re.search(r"\d+\.\d+(?:\.\d+)?", text)
        if m:
            out[tool] = m.group(0)
    return out


def probe_platform() -> dict[str, Any]:
    """Return a minimal hardware profile (no benchmarks). Cheap enough to run once/session."""
    profile: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu": _cpu(),
        "memory": _memory(),
        "gpu": _gpu(),
        "toolchains": _toolchains(),
        "slurm": {"available": shutil.which("sbatch") is not None},
    }
    return profile
