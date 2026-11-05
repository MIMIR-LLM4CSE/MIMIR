"""A run's profile as the map draws it: its shares in fractions of the baseline's metric,
so two runs' bars compare."""
from __future__ import annotations

import os

from ..._lib import profiling
from ..._lib.layout import opt_session_runs_dir, read_json

_KERNELS_SHOWN = 8


def scaled(name: str, run_id: str, value, baseline) -> dict | None:
    """*run_id*'s profile scaled by *value* / *baseline*; None when the run was not profiled.
    The sections taken apart (GPU, memory, counters, a kernel's) ride along even when the
    sampled CPU side could not run."""
    prof = read_json(profiling.profile_path(os.path.join(opt_session_runs_dir(name), run_id))) if run_id else None
    if not prof:
        return None
    numbers = isinstance(value, (int, float)) and isinstance(baseline, (int, float))
    scale = value / baseline if numbers and baseline else 1.0
    sections = {
        **({"gpu": _gpu(prof["gpu"], scale)} if prof.get("gpu") else {}),
        # Shares of the heap's peak, not of time: never scaled.
        **({"memory": {k: prof["memory"][k] for k in ("tool", "peak_mb", "files", "unavailable", "note")
                       if k in prof["memory"]}} if prof.get("memory") else {}),
        # Rates and percentages of peak, not shares of time: never scaled.
        **{k: prof[k] for k in ("counters", "kernel_detail") if prof.get(k)},
    }
    if prof.get("unavailable"):
        return {"unavailable": prof["unavailable"], "tool": prof.get("tool", ""), **sections}
    return {"tool": prof.get("tool", ""), "samples": prof.get("samples", 0), "case": prof.get("case", ""),
            "files": _shares(prof.get("files"), scale),
            "outside": round(sum((prof.get("outside") or {}).values()) * scale, 5),
            **({"functions": {k: {"self": round((v.get("self") or 0) * scale, 5),
                                  "total": round((v.get("total") or 0) * scale, 5),
                                  **({"line": v["line"]} if v.get("line") else {})}
                              for k, v in prof["functions"].items()}} if prof.get("functions") else {}),
            **sections,
            **({"note": prof["note"]} if prof.get("note") else {})}


def _shares(shares: dict | None, scale: float) -> dict:
    return {k: round(v * scale, 5) for k, v in (shares or {}).items()}


def _gpu(g: dict, scale: float) -> dict:
    """A profile's GPU side in fractions of the baseline's time, its costliest kernels."""
    if g.get("unavailable"):
        return {"unavailable": g["unavailable"], "tool": g.get("tool", "")}
    return {"tool": g.get("tool", ""),
            "busy": round((g.get("busy") or 0) * scale, 5), "idle": round((g.get("idle") or 0) * scale, 5),
            "memcpy": _shares(g.get("memcpy"), scale),
            "files": _shares(g.get("files"), scale),
            "kernels": {k: {**v, "share": round((v.get("share") or 0) * scale, 5)}
                        for k, v in list((g.get("kernels") or {}).items())[:_KERNELS_SHOWN]},
            # Sizes, not shares of time: never scaled.
            **{k: g[k] for k in ("memcpy_mb", "peak_device_mb") if g.get(k) is not None},
            **({"note": g["note"]} if g.get("note") else {})}
