"""Where a run's time goes, by the workspace's source files: one profiled execution of a
case, each sample given to the deepest workspace frame of its stack, by file, function
and line; a GPU's kernels and copies traced when the run uses one; on request, hardware
counters, one kernel's, or the allocations live at the heap's peak.

A profile is never a measurement: taken after the verdict, it moves nothing. When no
profiler can run here, the profile says why.
"""
from __future__ import annotations

import os
import re
import subprocess

from .....shared.atomic import write_json
from .....shared.atomic import read_json
from . import cpu, gpu, memory
from .cpu import attribute, choose, cprofile_files, is_python, perf_allowed, perf_stacks, pyspy_stacks
from .frames import Frame, resolver

PROFILE_DIR, PROFILE_FILE = "profile", "profile.json"
LEVELS = ("file", "function", "line", "counters", "kernel", "memory")
# Levels read from the sampled stacks every profile keeps; the others take a run of their own
# and add their section to the profile.
SAMPLED = ("file", "function", "line")
SECTION = {"counters": "counters", "kernel": "kernel_detail", "memory": "memory"}


def profile_path(run_dir: str) -> str:
    """Where a run's profile is kept: beside its measurements, never among them."""
    return os.path.join(run_dir, PROFILE_DIR, PROFILE_FILE)


def profile(argv: list[str], out_dir: str, root: str, files: list[str], *,
            timeout: float | None = None, case: str = "", level: str = "file", focus: str = "",
            wall_s: float | None = None, peak_mb: float | None = None,
            peak_gbytes_per_s: float | None = None) -> dict:
    """Run *argv* under the profiler *level* asks for, write ``profile.json`` in *out_dir*
    and return it; never raises. A sampled level keeps file, function and line together,
    and the GPU's kernels and copies (as shares of *wall_s*, the run's measured time) when
    the run uses one; ``counters``, ``kernel`` (of the kernel *focus*) and ``memory``
    (against *peak_mb*, the run's measured peak) add their section to the profile there."""
    log = os.path.join(out_dir, "profile.log")
    if level in SECTION:
        kept = read_json(os.path.join(out_dir, PROFILE_FILE)) or {"case": case}
        try:
            with open(log, "a") as fh:
                if level == "counters":
                    found = cpu.counters(argv, out_dir, fh, timeout, peak_gbytes_per_s)
                elif level == "memory":
                    found = memory.profile(argv, out_dir, resolver(root, os.getcwd(), files), fh, timeout, peak_mb)
                else:
                    found = _kernel(argv, out_dir, focus, kept, fh, timeout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            found = {"unavailable": f"{level} profiling failed: {exc}"}
        return _write(out_dir, {**kept, SECTION[level]: found})
    tool, why = choose(argv)
    result: dict = {"tool": tool, "case": case}
    if not tool:
        result["unavailable"] = why
    else:
        try:
            with open(log, "w") as fh:
                result.update(cpu.sample(tool, argv, out_dir, resolver(root, os.getcwd(), files), fh, timeout))
            if not result.get("samples"):
                result["unavailable"] = f"{tool} recorded nothing (see profile.log)"
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result["unavailable"] = f"{tool} failed: {exc}"
    # The GPU side is traced even when the CPU side could not be: the tools differ.
    traced = _gpu(argv, out_dir, root, files, list(result.get("outside") or {}), timeout, wall_s)
    if traced:
        result["gpu"] = traced
    if (not traced or traced.get("unavailable")) and any(
            re.search(r"cuda|hip|nvrtc|rocm", k, re.I) for k in result.get("outside", {})):
        result["note"] = "; ".join(filter(None, [result.get("note"),
                                                  "GPU time is not seen: the profile shows the run's CPU side"]))
    return _write(out_dir, result)


def _gpu(argv: list[str], out_dir: str, root: str, files: list[str], seen: list[str],
         timeout: float | None, wall_s: float | None) -> dict | None:
    """The GPU section of a run that uses one, or None for a run that does not."""
    vendor = gpu.gpu_vendor(argv, seen)
    if not vendor:
        return None
    tool, why = gpu.choose_gpu(vendor)
    if not tool:
        return {"vendor": vendor, "unavailable": why}
    try:
        with open(os.path.join(out_dir, "profile.log"), "a") as fh:
            return {"tool": tool, "vendor": vendor,
                    **gpu.trace(tool, argv, out_dir, root, files, fh, timeout, wall_s)}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"tool": tool, "vendor": vendor, "unavailable": f"{tool} failed: {exc}"}


def _kernel(argv: list[str], out_dir: str, name: str, kept: dict, log, timeout: float | None) -> dict:
    """One kernel's hardware counters, or why there are none."""
    known = list(((kept.get("gpu") or {}).get("kernels") or {}))
    if not name:
        return {"unavailable": "name the kernel with focus=" + (f"; known: {', '.join(known[:10])}" if known else "")}
    vendor = (kept.get("gpu") or {}).get("vendor") or gpu.gpu_vendor(argv, list(kept.get("outside") or {}))
    tool, why = gpu.choose_gpu(vendor, detail=True)
    if not tool:
        return {"unavailable": why}
    return gpu.kernel_detail(tool, argv, out_dir, name, log, timeout)


def _write(out_dir: str, result: dict) -> dict:
    try:
        write_json(os.path.join(out_dir, PROFILE_FILE), result, indent=2)
    except OSError:
        pass
    return result


def has(profile_: dict | None, level: str) -> bool:
    """Whether *profile_* already answers *level*, so no run is needed for it."""
    if not profile_ or profile_.get("unavailable"):
        return False
    return bool(profile_.get({"file": "files", "function": "functions", "line": "lines"}.get(level, level)))


def top(profile_: dict | None, n: int = 5, level: str = "file", focus: str = "") -> dict:
    """The profile in a few lines for the agent, at *level*: the *n* costliest workspace
    files (and the share spent outside the workspace), functions or lines, those of the
    file *focus* when given; or the hardware counters."""
    g = (profile_ or {}).get("gpu")
    if level in SECTION and profile_:
        found = profile_.get(SECTION[level])
        if not found:
            return {"unavailable": f"no {level} view yet: profile with level='{level}'"}
        return _memory_top(found, n, focus) if level == "memory" else found
    if not profile_ or profile_.get("unavailable"):
        return {"unavailable": (profile_ or {}).get("unavailable", "no profile"),
                **({"gpu": _gpu_top(g, n)} if g and level == "file" else {})}
    head = {"tool": profile_.get("tool"), **({"case": profile_["case"]} if profile_.get("case") else {})}
    note = {"note": profile_["note"]} if profile_.get("note") else {}
    if level == "file":
        files = list((profile_.get("files") or {}).items())[:n]
        return {**head, "files": dict(files),
                "outside": round(sum((profile_.get("outside") or {}).values()), 4),
                **({"gpu": _gpu_top(g, n)} if g else {}), **note}
    key, sep = ("functions", "::") if level == "function" else ("lines", ":")
    rows = profile_.get(key)
    if not rows:
        why = ("cProfile keeps no time per line; install py-spy for it" if profile_.get("tool") == "cprofile"
               else f"this profile keeps no time per {level}; profile again with op='profile'")
        return {"unavailable": why}
    picked = [(k, v) for k, v in rows.items() if not focus or k.startswith(focus + sep)]
    if not picked:
        return {**head, level + "s": {}, "note": f"no {level} of {focus} took any of the profiled time"}
    return {**head, **({"focus": focus} if focus else {}), level + "s": dict(picked[:n]), **note}


def _memory_top(m: dict, n: int, focus: str) -> dict:
    """Where the peak's bytes were allocated: the costliest files and the share outside the
    workspace, or the functions and lines of the file *focus*."""
    if m.get("unavailable"):
        return {"unavailable": m["unavailable"]}
    head = {"tool": m.get("tool"), "peak_mb": m.get("peak_mb"), **({"note": m["note"]} if m.get("note") else {})}
    if not focus:
        return {**head, "files": dict(list((m.get("files") or {}).items())[:n]),
                "outside": round(sum((m.get("outside") or {}).values()), 4)}
    fns = {k: v for k, v in (m.get("functions") or {}).items() if k.startswith(focus + "::")}
    lines = {k: v for k, v in (m.get("lines") or {}).items() if k.startswith(focus + ":")}
    return {**head, "focus": focus, "functions": dict(list(fns.items())[:n]), "lines": dict(list(lines.items())[:n])}


def _gpu_top(g: dict, n: int) -> dict:
    """The GPU side in a few lines: busy, copies and idle as shares of the run's time, and
    the *n* costliest kernels."""
    if g.get("unavailable"):
        return {"unavailable": g["unavailable"]}
    return {"tool": g.get("tool"), "busy": g.get("busy"), "memcpy": g.get("memcpy") or {}, "idle": g.get("idle"),
            **({"memcpy_mb": g["memcpy_mb"]} if g.get("memcpy_mb") else {}),
            **({"peak_device_mb": g["peak_device_mb"]} if g.get("peak_device_mb") is not None else {}),
            "kernels": {k: v for k, v in list((g.get("kernels") or {}).items())[:n]},
            **({"note": g["note"]} if g.get("note") else {})}


__all__ = ["Frame", "LEVELS", "PROFILE_DIR", "PROFILE_FILE", "SAMPLED", "SECTION", "attribute", "choose", "cprofile_files",
           "has", "is_python", "perf_allowed", "perf_stacks", "profile", "profile_path", "pyspy_stacks",
           "resolver", "top"]
