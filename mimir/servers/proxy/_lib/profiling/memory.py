"""Where a run's memory goes: the allocations live at its heap's peak, each given to the
deepest workspace frame of the stack that made it, by file, function and line, as shares
of the bytes live then.

memray (or tracemalloc, near the peak) for a Python command; heaptrack, else valgrind's
massif, for a compiled one, or for a Python one whose traced heap is under half the
measured peak (its child processes, which only they follow), read through one reader.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from typing import Callable, TextIO

from .. import proc_run
from .cpu import attribute, is_python
from .frames import Frame, fortran_name, is_fortran

_NODE_RE = re.compile(r"^(?P<indent> *)n(?P<kids>\d+): (?P<bytes>\d+) (?P<rest>.*)$")
_SITE_RE = re.compile(r"^0x[0-9A-Fa-f]+: (?P<func>.*?)(?: \((?P<loc>[^()]*)\))?$")

# Run as the profiled Python's own: read memray's capture at its high-water mark.
_MEMRAY_READER = """\
import json, sys
from memray import FileReader
reader = FileReader(sys.argv[1])
stacks = []
for rec in reader.get_high_watermark_allocation_records(merge_threads=True):
    try:
        trace = rec.hybrid_stack_trace()
    except Exception:
        trace = rec.stack_trace()
    stacks.append([rec.size, [[f, p, l] for f, p, l in reversed(trace)]])
json.dump({"peak": reader.metadata.peak_memory, "stacks": stacks}, open(sys.argv[2], "w"))
"""

# Run in place of the profiled script: tracemalloc snapshots taken as the traced memory
# grows, the largest kept, so the stacks are those live near the peak.
_TRACEMALLOC_BOOT = """\
import atexit, json, os, runpy, sys, threading, time, tracemalloc
out, sys.argv = sys.argv[1], sys.argv[2:]
tracemalloc.start(32)
best = {"cur": 0, "snap": None}
def watch():
    while True:
        cur, _ = tracemalloc.get_traced_memory()
        if cur > best["cur"] * 1.05 + 2**20:
            best.update(cur=cur, snap=tracemalloc.take_snapshot())
        time.sleep(0.01)
threading.Thread(target=watch, daemon=True).start()
def dump():
    cur, peak = tracemalloc.get_traced_memory()
    snap = best["snap"] if best["cur"] >= cur else tracemalloc.take_snapshot()
    stacks = [[s.size, [["", f.filename, f.lineno] for f in s.traceback]]
              for s in snap.statistics("traceback")] if snap else []
    json.dump({"peak": peak, "stacks": stacks}, open(out, "w"))
atexit.register(dump)
if sys.argv[0] == "-m":
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _has_memray(python: str) -> bool:
    try:
        return subprocess.run([python, "-c", "import memray"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose(argv: list[str]) -> tuple[str, str]:
    """The allocation profiler for *argv* here, or "" with the reason none fits."""
    if is_python(argv):
        if _has_memray(argv[0]):
            return "memray", ""
        if len(argv) > 1 and argv[1] != "-c":
            return "tracemalloc", ""
        return "", "memray is not installed (pip install memray) and tracemalloc cannot run a `-c` command"
    return choose_native()


def choose_native() -> tuple[str, str]:
    """The allocation profiler that follows a command's processes, whatever they run."""
    if shutil.which("heaptrack") and shutil.which("heaptrack_print"):
        return "heaptrack", ""
    if shutil.which("valgrind"):
        return "massif", ""
    return "", "neither heaptrack nor valgrind is installed"


# ── reading what a profiler wrote ────────────────────────────────────────────

def massif_peak(text: str) -> tuple[int, list[tuple[list[Frame], int]]]:
    """A massif file's peak snapshot (else its largest detailed one): the heap's bytes
    then, and each allocation site's stack root first with its bytes."""
    best: tuple[int, list[str]] = (-1, [])
    for block in re.split(r"^#-+\n(?=snapshot=)", text, flags=re.M):
        heap = re.search(r"^mem_heap_B=(\d+)", block, re.M)
        tree = re.search(r"^heap_tree=(\w+)", block, re.M)
        if not heap or not tree or tree.group(1) == "empty":
            continue
        size, lines = int(heap.group(1)), block[tree.end():].splitlines()
        if tree.group(1) == "peak":
            best = (size, lines)
            break
        if size > best[0]:
            best = (size, lines)
    return max(best[0], 0), _massif_tree(best[1])


def _massif_tree(lines: list[str]) -> list[tuple[list[Frame], int]]:
    path: list[Frame | None] = []
    stacks: list[tuple[list[Frame], int]] = []
    for line in lines:
        node = _NODE_RE.match(line)
        if not node:
            continue
        depth, kids, size = len(node.group("indent")), int(node.group("kids")), int(node.group("bytes"))
        del path[depth:]
        site = _SITE_RE.match(node.group("rest"))
        frame = None
        if "below massif's threshold" in node.group("rest"):
            frame = Frame("small allocations")
        elif site and depth:
            loc = site.group("loc") or ""
            if loc.startswith("in "):
                frame = Frame(loc[3:])
            else:
                where, _, lineno = loc.rpartition(":")
                frame = Frame(where, fortran_name(site.group("func"), where), int(lineno) if lineno.isdigit() else 0)
        path.append(frame)
        if kids == 0 and depth and size:
            # Innermost first below the allocation functions: the stack is that, reversed.
            stacks.append(([f for f in reversed(path) if f is not None], size))
    return stacks


def json_stacks(data: dict) -> tuple[int, list[tuple[list[Frame], int]]]:
    """The readers' own format: ``{"peak": bytes, "stacks": [[bytes, [[func, file, line], ...]]]}``,
    frames root first."""
    stacks = [([Frame(where or "", fortran_name(func or "", where or ""), int(line or 0)) for func, where, line in frames],
               int(size)) for size, frames in data.get("stacks") or []]
    return int(data.get("peak") or 0), stacks


# ── taking one ───────────────────────────────────────────────────────────────

def profile(argv: list[str], out_dir: str, resolve: Callable[[str], str | None], log: TextIO,
            timeout: float | None, measured_mb: float | None = None) -> dict:
    """Run *argv* once under the allocation profiler that fits and say where its peak's
    bytes were allocated; *measured_mb* is the run's measured peak, for comparison."""
    tool, why = choose(argv)
    if not tool:
        return {"unavailable": why}
    mem_dir = os.path.join(out_dir, "memory")
    os.makedirs(mem_dir, exist_ok=True)
    peak, stacks = _take(tool, argv, mem_dir, log, timeout)
    notes = []
    if is_python(argv) and measured_mb and peak / 2**20 < 0.5 * measured_mb:
        native, why = choose_native()
        if native:
            more = _take(native, argv, os.path.join(mem_dir, native), log, timeout)
            if more[0] > peak:
                notes.append(f"{tool} traced {peak / 2**20:.0f} of the {measured_mb:.0f} MB measured, "
                             f"so {native} profiled the processes the command starts")
                tool, (peak, stacks) = native, more
        else:
            notes.append(f"the processes the command starts are not followed: {why}")
    if not stacks:
        return {"tool": tool, "unavailable": f"{tool} recorded no allocation (see profile.log)"}
    found = attribute(stacks, resolve)
    found.pop("samples", None)
    result = {"tool": tool, "peak_mb": round(peak / 2**20, 1), **found}
    if tool == "tracemalloc":
        notes.append("tracemalloc: Python's allocations near the peak, by file and line")
    if tool == "massif":
        notes.append("massif runs the program on a simulated CPU, many times slower")
    if any(is_fortran(f) for f in found.get("files") or {}):
        notes.append("Fortran's automatic arrays live on the stack, not the heap, and are not counted")
    if measured_mb and peak and peak / 2**20 < 0.5 * measured_mb:
        notes.append(f"the heap traced holds {peak / 2**20:.0f} of the {measured_mb:.0f} MB measured: the rest "
                     "is code, libraries, the stack, or a process this profile does not follow")
    if notes:
        result["note"] = "; ".join(notes)
    return result


def _take(tool: str, argv: list[str], mem_dir: str, log: TextIO,
          timeout: float | None) -> tuple[int, list[tuple[list[Frame], int]]]:
    os.makedirs(mem_dir, exist_ok=True)
    if tool in ("memray", "tracemalloc"):
        out = os.path.join(mem_dir, "peak.json")
        if tool == "memray":
            capture = os.path.join(mem_dir, "memray.bin")
            proc_run.run([argv[0], "-m", "memray", "run", "--native", "--force", "-o", capture, *argv[1:]],
                         stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            proc_run.run([argv[0], "-c", _MEMRAY_READER, capture, out],
                         stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        else:
            proc_run.run([argv[0], "-c", _TRACEMALLOC_BOOT, out, *argv[1:]],
                         stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        try:
            with open(out) as fh:
                return json_stacks(json.load(fh))
        except (OSError, ValueError):
            return 0, []                 # the run failed under it: profile.log says why
    if tool == "heaptrack":
        proc_run.run(["heaptrack", "-o", os.path.join(mem_dir, "heaptrack"), *argv],
                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        for i, data in enumerate(sorted(glob.glob(os.path.join(mem_dir, "heaptrack*.*z*")))):
            proc_run.run(["heaptrack_print", "-f", data, "--print-massif", os.path.join(mem_dir, f"massif.out.{i}"),
                          "--massif-detailed-freq", "1"], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    else:
        proc_run.run(["valgrind", "--tool=massif", "--trace-children=yes", "--threshold=0.5", "--detailed-freq=1",
                      f"--massif-out-file={os.path.join(mem_dir, 'massif.out.%p')}", *argv],
                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    # One file per process the run started: the one whose peak is largest is the run's.
    best: tuple[int, list] = (0, [])
    for path in glob.glob(os.path.join(mem_dir, "massif.out.*")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            peak, stacks = massif_peak(fh.read())
        if peak > best[0]:
            best = (peak, stacks)
    return best
