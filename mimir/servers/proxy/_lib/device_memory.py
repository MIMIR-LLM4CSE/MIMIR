"""The GPU memory in use now, in MB: per process for NVIDIA (``nvidia-smi``), for the
whole device for AMD (``rocm-smi``); read by a GPU trace and by a measured region."""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict

_NVIDIA = {"cuda", "cudart", "nvrtc"}
_AMD = {"amdhip64", "hiprtc"}


def vendor_of(runtimes: set[str] | list[str]) -> str:
    """"nvidia" or "amd" for a run that mapped these runtimes (``proc_run``), else ""."""
    found = set(runtimes)
    return "nvidia" if found & _NVIDIA else "amd" if found & _AMD else ""


def _number(text: str) -> float:
    return float((text or "0").replace(",", "").strip() or 0)


def sample_device(vendor: str) -> dict[str, float] | None:
    """GPU memory in use now, in MB: per process for NVIDIA, per device for AMD."""
    try:
        if vendor == "nvidia":
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            used: defaultdict[str, float] = defaultdict(float)
            for line in out.splitlines():
                pid, _, mb = line.partition(",")
                if mb.strip().replace(".", "", 1).isdigit():
                    used[pid.strip()] += float(mb)
            return dict(used)
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=5).stdout
        cards = json.loads(out or "{}")
        return {"device": sum(_number(c.get("VRAM Total Used Memory (B)", "0")) for c in cards.values()
                              if isinstance(c, dict)) / 2**20}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
