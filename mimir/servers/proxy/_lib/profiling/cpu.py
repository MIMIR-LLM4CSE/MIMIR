"""The CPU side of a profile: py-spy (``--native``: the C/C++ frames of a Python process
too; ``--subprocesses``: the Python processes it starts) or cProfile for a Python command,
perf for a compiled one, each sample given to the deepest workspace frame of its stack, by
file, function and line; and perf's hardware counters for a compiled command, on request.

A Python harness that runs the code in a program py-spy cannot read (a compiled solver it
starts) leaves py-spy with the harness waiting: when py-spy saw less than half the run,
perf profiles it again, and the profile that puts more seconds in the workspace is kept."""
from __future__ import annotations

import os
import pstats
import re
import shutil
import subprocess
import time
from collections import Counter
from typing import Callable, TextIO

from .. import proc_run
from .frames import Frame, fortran_name, outside

FUNCTIONS_KEPT, LINES_KEPT = 50, 200

_PYTHON_RE = re.compile(r"^(python|pypy)[\d.]*$")
_PYSPY_FRAME_RE = re.compile(r"^(?P<name>.*?) \((?P<where>[^()]*)\)$")
# The root frames ``--subprocesses`` gives each stack: the process, then its parents'.
_PYSPY_PROCESS_RE = re.compile(r'^(?:process \d+:"(?:[^"\\]|\\.)*";)+')
PYSPY_HZ, PERF_HZ = 250, 499
# A py-spy profile that saw less than this share of the run missed where it went.
SEEN_ENOUGH = 0.5
_HEX_RE = re.compile(r"^\s*([0-9a-f]+)\s+(?P<sym>\S.*?)?\s*\((?P<dso>[^()]*)\)\s*$")
_SRCLINE_RE = re.compile(r"^\s+(?P<file>[^\s:][^:]*):(?P<line>\d+|\?\?)\s*$")
_PERF_EVENTS = ("cycles", "instructions", "cache-references", "cache-misses", "branches", "branch-misses")
_LLC_EVENTS = ("LLC-loads", "LLC-load-misses")


def is_python(argv: list[str]) -> bool:
    """Whether a command runs Python: an interpreter, or a script it runs."""
    return bool(argv) and (bool(_PYTHON_RE.match(os.path.basename(argv[0]))) or argv[0].endswith(".py"))


def perf_allowed() -> tuple[bool, str]:
    """Whether perf may profile here, and why not when it may not."""
    if not shutil.which("perf"):
        return False, "perf is not installed"
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True, ""
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as fh:
            level = int(fh.read().strip())
    except (OSError, ValueError):
        return True, ""
    if level > 2:
        return False, (f"perf is blocked here (kernel.perf_event_paranoid = {level}); an admin can "
                       "allow it with `sudo sysctl kernel.perf_event_paranoid=1`")
    return True, ""


def choose(argv: list[str]) -> tuple[str, str]:
    """The profiler for *argv* here, or "" with the reason none fits."""
    if is_python(argv):
        if shutil.which("py-spy"):
            return "py-spy", ""
        if len(argv) > 1 and argv[1] != "-c":
            return "cprofile", ""
        return "", "py-spy is not installed and cProfile cannot run a `-c` command"
    ok, why = perf_allowed()
    return ("perf", "") if ok else ("", why)


# ── reading what a profiler wrote ────────────────────────────────────────────

def pyspy_stacks(text: str) -> list[tuple[list[Frame], int]]:
    """py-spy raw output: each stack root first, as its frames, and its count."""
    out = []
    for line in text.splitlines():
        stack, _, count = line.rpartition(" ")
        stack = _PYSPY_PROCESS_RE.sub("", stack)
        if not stack or not count.isdigit():
            continue
        frames = []
        for frame in stack.split(";"):
            m = _PYSPY_FRAME_RE.match(frame.strip())
            name, where = (m.group("name"), m.group("where")) if m else ("", "")
            path, sep, lineno = where.rpartition(":")
            if not (sep and lineno.isdigit()):
                path, lineno = where, "0"
            frames.append(Frame(path, "" if name.startswith("0x") else fortran_name(name, path), int(lineno)))
        out.append((frames, int(count)))
    return out


def perf_stacks(text: str) -> list[tuple[list[Frame], int]]:
    """``perf script -F ip,sym,dso,srcline`` output: each sample's frames root first, each
    its source file and line when perf knew them, else its library."""
    out: list[tuple[list[Frame], int]] = []
    frames: list[Frame] = []
    for line in text.splitlines() + [""]:
        if not line.strip():
            if frames:
                out.append(([f._replace(func=fortran_name(f.func, f.where)) for f in reversed(frames)], 1))
                frames = []
            continue
        src = _SRCLINE_RE.match(line)
        if src and frames and not _HEX_RE.match(line):
            if "??" not in (src.group("file"), src.group("line")):
                frames[-1] = frames[-1]._replace(where=src.group("file"), line=int(src.group("line")))
            continue
        hexed = _HEX_RE.match(line)
        if hexed:
            sym = re.sub(r"\+0x[0-9a-f]+$", "", hexed.group("sym") or "")
            frames.append(Frame(hexed.group("dso"), "" if sym == "[unknown]" else sym))
    return out


def attribute(stacks: list[tuple[list[Frame], int]], resolve: Callable[[str], str | None]) -> dict:
    """Shares of the samples per workspace file, function and line (the deepest workspace
    frame of each stack; a function's total counts every stack it is on, once) and, for
    stacks with none, per outside package or library. A native frame lands by its symbol
    when its file does not (``resolve.symbol``), opened where its function is defined."""
    files: Counter = Counter()
    own: Counter = Counter()
    under: Counter = Counter()
    lines: Counter = Counter()
    at: Counter = Counter()
    out: Counter = Counter()
    defs: dict[str, int] = {}
    total = 0
    by_symbol = getattr(resolve, "symbol", None)

    def home_of(f: Frame) -> str | None:
        home = resolve(f.where)
        if home or not by_symbol:
            return home
        hit = by_symbol(f.func, f.where)
        if hit:
            defs[f"{hit[0]}::{f.func}"] = hit[1]
        return hit[0] if hit else None
    for frames, count in stacks:
        total += count
        homes = [(home, f) for f in frames if (home := home_of(f))]
        if not homes:
            out[outside(frames[-1].where if frames else "")] += count
            continue
        home, deepest = homes[-1]
        files[home] += count
        if deepest.func:
            own[f"{home}::{deepest.func}"] += count
            if deepest.line:
                at[(f"{home}::{deepest.func}", deepest.line)] += count
        if deepest.line:
            lines[f"{home}:{deepest.line}"] += count
        for key in {f"{h}::{f.func}" for h, f in homes if f.func}:
            under[key] += count
    if not total:
        return {"samples": 0, "files": {}, "outside": {}}
    return {"samples": total,
            "files": {f: round(n / total, 4) for f, n in files.most_common()},
            "functions": _functions(own, under, total, {**defs, **_costliest_lines(at)}),
            "lines": {k: round(n / total, 4) for k, n in lines.most_common(LINES_KEPT)},
            "outside": {k: round(n / total, 4) for k, n in out.most_common(12)}}


def cprofile_files(path: str, resolve: Callable[[str], str | None]) -> dict:
    """A cProfile dump's self time per workspace file and function (with its cumulative
    time), the rest per outside package."""
    stats = pstats.Stats(path).stats  # type: ignore[attr-defined]
    files: Counter = Counter()
    own: Counter = Counter()
    under: Counter = Counter()
    out: Counter = Counter()
    defs: dict[str, int] = {}
    for (filename, line, func), (_cc, _nc, tottime, cumtime, _callers) in stats.items():
        home = resolve(filename) if filename and filename[0] not in "<~" else None
        if home:
            files[home] += tottime
            own[f"{home}::{func}"] += tottime
            under[f"{home}::{func}"] += cumtime
            if line:
                defs[f"{home}::{func}"] = line
        else:
            out["built-ins" if filename == "~" else "python" if filename.startswith("<") else outside(filename)] += tottime
    total = sum(files.values()) + sum(out.values())
    if not total:
        return {"samples": 0, "files": {}, "outside": {}}
    return {"samples": len(stats),
            "files": {f: round(t / total, 4) for f, t in files.most_common() if t / total >= 5e-5},
            "functions": _functions(own, under, total, defs),
            "outside": {k: round(t / total, 4) for k, t in out.most_common(12) if t / total >= 5e-5},
            "note": "cProfile: self time per file and function, without the native code it "
                    "calls, and no time per line"}


def _costliest_lines(at: Counter) -> dict[str, int]:
    """Each function's line that took the most samples."""
    best: dict[str, int] = {}
    for (fn, line), _n in sorted(at.items(), key=lambda kv: kv[1]):
        best[fn] = line
    return best


def _functions(own: Counter, under: Counter, total: float, lines: dict[str, int] | None = None) -> dict:
    """The costliest functions by self time and by total time, each with both, and the
    line to open it at when known."""
    keep = dict.fromkeys([k for k, _ in own.most_common(FUNCTIONS_KEPT)]
                         + [k for k, _ in under.most_common(FUNCTIONS_KEPT)])
    rows = {k: {"self": round(own[k] / total, 4), "total": round(under[k] / total, 4),
                **({"line": lines[k]} if lines and k in lines else {})} for k in keep}
    return {k: v for k, v in sorted(rows.items(), key=lambda kv: (-kv[1]["self"], -kv[1]["total"]))
            if v["self"] >= 5e-5 or v["total"] >= 5e-5}


def perf_counts(text: str) -> dict[str, tuple[float, str]]:
    """``perf stat -x,`` output: each event's count and unit, summed over the PMUs an
    uncore event spans; what perf could not count is left out."""
    seen: dict[str, tuple[float, str]] = {}
    for line in text.splitlines():
        cols = line.split(",")
        if len(cols) < 3:
            continue
        try:
            value = float(cols[0])
        except ValueError:
            continue                    # <not supported>, <not counted>, comments
        name = re.sub(r"^(uncore_imc)_\d+/", r"\1/", cols[2].split(":")[0])
        seen[name] = (seen.get(name, (0.0, ""))[0] + value, cols[1])
    return seen


def perf_stat(text: str) -> dict:
    """``perf stat -x,`` output read into instructions per cycle and miss rates."""
    seen = {k: v for k, (v, _unit) in perf_counts(text).items()}

    def ratio(a: str, b: str) -> float | None:
        return round(seen[a] / seen[b], 4) if seen.get(a) is not None and seen.get(b) else None
    counters = {"ipc": ratio("instructions", "cycles"),
                "cache_miss_rate": ratio("cache-misses", "cache-references"),
                "branch_miss_rate": ratio("branch-misses", "branches"),
                "llc_miss_rate": ratio("LLC-load-misses", "LLC-loads")}
    return {k: v for k, v in counters.items() if v is not None}


def bandwidth(core: str, uncore: str, elapsed_s: float, peak_gbytes_per_s: float | None) -> dict:
    """DRAM traffic per second: the memory controllers' bytes when *uncore* counted them
    (machine-wide), else last-level cache load misses times a 64-byte line, a lower
    bound; and its share of the declared peak."""
    ctrl = perf_counts(uncore)
    moved = sum(v * (2**20 if unit == "MiB" else 64) for name, (v, unit) in ctrl.items()
                if re.search(r"cas_count_(read|write)|data_(reads|writes)", name))
    if moved:
        how = "memory controllers, whole machine"
    else:
        misses = perf_counts(core).get("LLC-load-misses")
        if not misses or elapsed_s <= 0:
            return {}
        moved, how = misses[0] * 64, "last-level cache load misses x 64 B: a lower bound, without writes or prefetches"
    rate = moved / elapsed_s / 1e9
    return {"dram_gbytes_per_s": round(rate, 2), "dram_source": how,
            **({"bandwidth_fraction": round(rate / peak_gbytes_per_s, 3)} if peak_gbytes_per_s else {})}


def _uncore_events() -> tuple[str, ...]:
    """The memory controllers' read and write events, when perf may count the whole
    machine here (root, or kernel.perf_event_paranoid at 0 or below) and knows them."""
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as fh:
            allowed = int(fh.read().strip()) <= 0
    except (OSError, ValueError):
        allowed = False
    if not (allowed or (hasattr(os, "geteuid") and os.geteuid() == 0)):
        return ()
    try:
        listing = subprocess.run(["perf", "list"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    for reads, writes in (("cas_count_read", "cas_count_write"), ("data_reads", "data_writes")):
        if reads in listing:
            return f"uncore_imc/{reads}/", f"uncore_imc/{writes}/"
    return ()


# ── taking one ───────────────────────────────────────────────────────────────

def sample(tool: str, argv: list[str], out_dir: str, resolve: Callable[[str], str | None],
           log: TextIO, timeout: float | None, wall_s: float | None = None) -> dict:
    """Run *argv* once under *tool* and attribute its samples. A py-spy profile that saw
    less than half of *wall_s* (the run's measured time), or put more than half of it on
    the command's own script (the harness, whose native callee it could not read), is taken
    again by perf, and the one that puts more seconds in the workspace's code is kept."""
    if tool == "cprofile":
        dump = os.path.join(out_dir, "profile.pstats")
        proc_run.run([argv[0], "-m", "cProfile", "-o", dump, *argv[1:]], stdout=log,
                     stderr=subprocess.STDOUT, timeout=timeout)
        return cprofile_files(dump, resolve)
    if tool != "py-spy":
        return _perf_sample(argv, out_dir, resolve, log, timeout)
    found = _pyspy(argv, out_dir, resolve, log, timeout)
    script = resolve(argv[1]) if len(argv) > 1 else None
    seen = found.get("samples", 0) / PYSPY_HZ
    if wall_s and seen < SEEN_ENOUGH * wall_s:
        missed = (f"py-spy saw {seen:.1f} s of the run's {wall_s:.1f} s: the rest ran in a process "
                  "it cannot read (a compiled program the command starts)")
    elif script and (found.get("files") or {}).get(script, 0) > SEEN_ENOUGH:
        missed = (f"py-spy left {found['files'][script]:.0%} of the run on {script}, the command's own "
                  "script: in native code it calls that py-spy could not place")
    else:
        return found
    ok, why = perf_allowed()
    if not ok:
        return {**found, "note": f"{missed}; perf would see it, but {why}"}
    native = _perf_sample(argv, out_dir, resolve, log, timeout)
    if _workspace_s(native, PERF_HZ, script) <= _workspace_s(found, PYSPY_HZ, script):
        return {**found, "note": f"{missed}; perf saw no more of the workspace"}
    return {**native, "tool": "perf",
            "note": "; ".join(filter(None, [f"{missed}, so perf profiled it: Python's frames are "
                                            "the interpreter's there", native.get("note")]))}


def _workspace_s(found: dict, hz: int, but: str | None = None) -> float:
    """The seconds a sampled profile puts in the workspace's files, *but* one (the harness)."""
    return found.get("samples", 0) / hz * sum(v for k, v in (found.get("files") or {}).items() if k != but)


def _pyspy(argv: list[str], out_dir: str, resolve: Callable[[str], str | None],
           log: TextIO, timeout: float | None) -> dict:
    raw = os.path.join(out_dir, "profile.raw")
    base = ["py-spy", "record", "--format", "raw", "--rate", str(PYSPY_HZ), "-o", raw]
    # --native is not supported everywhere, and an older py-spy lacks the others.
    for flags in (["--native", "--subprocesses", "--full-filenames"], ["--subprocesses", "--full-filenames"], []):
        proc_run.run([*base, *flags, "--", *argv], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if os.path.isfile(raw):
            break
    with open(raw, encoding="utf-8", errors="replace") as rf:
        return attribute(pyspy_stacks(rf.read()), resolve)


def _perf_sample(argv: list[str], out_dir: str, resolve: Callable[[str], str | None],
                 log: TextIO, timeout: float | None) -> dict:
    """perf's profile of *argv* and the children it starts."""
    stacks = _perf(argv, out_dir, log, timeout, "fp")
    if stacks and shallow(stacks) > SHALLOW:
        # No frame pointers (an optimised build): unwind from the stack's own copy instead.
        deeper = _perf(argv, out_dir, log, timeout, "dwarf")
        if deeper and shallow(deeper) < shallow(stacks):
            stacks = deeper
    result = attribute(stacks, resolve)
    notes = []
    if stacks and shallow(stacks) > SHALLOW:
        notes.append("stacks are cut short, so a function's total time is understated: build with "
                     "-fno-omit-frame-pointer")
    binary = os.path.basename(argv[0])
    if not is_python(argv) and (result.get("outside") or {}).get(binary, 0) > 0.3:
        notes.append(f"{binary} has no debug information, so its time is not on its source files: "
                     "add -g to the build (it does not change the generated code)")
    if notes:
        result["note"] = "; ".join(notes)
    return result


# A profile whose stacks are mostly a single frame was unwound without frame pointers.
SHALLOW = 0.8


def shallow(stacks: list[tuple[list[Frame], int]]) -> float:
    """The share of samples whose stack is a single frame."""
    total = sum(n for _, n in stacks)
    return sum(n for frames, n in stacks if len(frames) <= 1) / total if total else 0.0


def _perf(argv: list[str], out_dir: str, log: TextIO, timeout: float | None, unwind: str) -> list:
    """perf's stacks of one run of *argv*, unwound by frame pointer ("fp") or DWARF."""
    data = os.path.join(out_dir, "perf.data")
    graph = ["-g"] if unwind == "fp" else ["--call-graph", "dwarf,16384"]
    proc_run.run(["perf", "record", *graph, "-F", str(PERF_HZ), "-o", data, "--", *argv],
                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    script = proc_run.run(["perf", "script", "-i", data, "-F", "ip,sym,dso,srcline"],
                          stdout=subprocess.PIPE, stderr=log, text=True, timeout=timeout)
    return perf_stacks(script.stdout or "")


def counters(argv: list[str], out_dir: str, log: TextIO, timeout: float | None,
             peak_gbytes_per_s: float | None = None) -> dict:
    """The hardware counters of one run of a compiled *argv*, its DRAM traffic against
    *peak_gbytes_per_s*, or why there are none."""
    if is_python(argv):
        return {"unavailable": "counters are for compiled code; this command runs Python"}
    ok, why = perf_allowed()
    if not ok:
        return {"unavailable": why}
    stat, ustat = os.path.join(out_dir, "perf.stat"), os.path.join(out_dir, "perf.uncore")
    cmd = ["perf", "stat", "-x,", "-o", stat, "-e", ",".join(_PERF_EVENTS + _LLC_EVENTS), "--", *argv]
    uncore = _uncore_events()
    if uncore:
        # The memory controllers belong to the machine, not the process: an outer,
        # machine-wide perf around the per-process one, over the same single run.
        cmd = ["perf", "stat", "-a", "-x,", "-o", ustat, "-e", ",".join(uncore), "--", *cmd]
    start = time.monotonic()
    proc_run.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    elapsed = time.monotonic() - start
    with open(stat, encoding="utf-8", errors="replace") as fh:
        core = fh.read()
    found = perf_stat(core)
    if not found:
        return {"unavailable": "perf stat counted nothing here (no hardware counters on this machine?)"}
    controllers = ""
    if uncore and os.path.isfile(ustat):
        with open(ustat, encoding="utf-8", errors="replace") as fh:
            controllers = fh.read()
    return {"tool": "perf stat", **found, **bandwidth(core, controllers, elapsed, peak_gbytes_per_s)}
