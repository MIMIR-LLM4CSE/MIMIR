"""Where a run's time goes, by the workspace's source files: one profiled execution of a
case, attributed sample by sample to the deepest workspace frame of its stack.

py-spy (``--native``: the C/C++ frames of a Python process too) or cProfile for a Python
command, perf for a compiled one; when none can run here, a profile that says why. A
profile is never a measurement: it is taken after the verdict and moves nothing.
"""
from __future__ import annotations

import os
import posixpath
import pstats
import re
import shutil
import subprocess
from collections import Counter
from typing import Callable

from ....shared.atomic import write_json
from . import proc_run

PROFILE_DIR, PROFILE_FILE = "profile", "profile.json"


def profile_path(run_dir: str) -> str:
    """Where a run's profile is kept: beside its measurements, never among them."""
    return os.path.join(run_dir, PROFILE_DIR, PROFILE_FILE)
_PYTHON_RE = re.compile(r"^(python|pypy)[\d.]*$")
_PYSPY_FRAME_RE = re.compile(r"^(?P<name>.*?) \((?P<where>[^()]*)\)$")
_HEX_RE = re.compile(r"^\s*([0-9a-f]+)\s+(\S.*?)?\s*\((?P<dso>[^()]*)\)\s*$")
_SRCLINE_RE = re.compile(r"^\s+(?P<file>[^\s:][^:]*):(?P<line>\d+|\?\?)\s*$")


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


# ── where a frame's file is ──────────────────────────────────────────────────

def resolver(root: str, cwd: str, files: list[str]) -> Callable[[str], str | None]:
    """A function giving a frame's file as a workspace path, or None outside it: absolute
    under *root*, relative to *cwd*, else the one workspace file it is a suffix of."""
    root = os.path.realpath(root)
    known = set(files)
    by_base: dict[str, list[str]] = {}
    for f in files:
        by_base.setdefault(posixpath.basename(f), []).append(f)

    def resolve(path: str) -> str | None:
        if not path:
            return None
        full = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
        if full.startswith(root + os.sep):
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in known or not known:
                return rel
        clean = path.replace(os.sep, "/").lstrip("./")
        hits = [f for f in by_base.get(posixpath.basename(clean), [])
                if f == clean or clean.endswith("/" + f) or f.endswith("/" + clean)]
        return hits[0] if len(hits) == 1 else None
    return resolve


def _outside(path_or_lib: str) -> str:
    """What a frame outside the workspace belongs to: a package, a library, Python itself."""
    p = path_or_lib.replace(os.sep, "/")
    m = re.search(r"(?:site|dist)-packages/([^/]+)", p)
    if m:
        return m.group(1).split(".")[0]
    base = posixpath.basename(p)
    if re.match(r"lib.+\.so", base) or ".so" in base:
        return re.sub(r"^lib|\.so.*$|\.cpython.*$", "", base) or base
    if p.endswith(".py"):
        return "python"
    return base or "unknown"


# ── reading what a profiler wrote ────────────────────────────────────────────

def pyspy_stacks(text: str) -> list[tuple[list[str], int]]:
    """py-spy raw output: each stack root first, as the frames' file (or library), and its count."""
    out = []
    for line in text.splitlines():
        stack, _, count = line.rpartition(" ")
        if not stack or not count.isdigit():
            continue
        frames = []
        for frame in stack.split(";"):
            m = _PYSPY_FRAME_RE.match(frame.strip())
            where = m.group("where") if m else ""
            frames.append(where.rsplit(":", 1)[0] if re.search(r":\d+$", where) else where)
        out.append((frames, int(count)))
    return out


def perf_stacks(text: str) -> list[tuple[list[str], int]]:
    """``perf script -F ip,sym,dso,srcline`` output: each sample's frames root first, each
    its source file when perf knew it, else its library."""
    out: list[tuple[list[str], int]] = []
    frames: list[str] = []
    for line in text.splitlines() + [""]:
        if not line.strip():
            if frames:
                out.append((list(reversed(frames)), 1))
                frames = []
            continue
        src = _SRCLINE_RE.match(line)
        if src and frames and not _HEX_RE.match(line):
            if "??" not in (src.group("file"), src.group("line")):
                frames[-1] = src.group("file")
            continue
        hexed = _HEX_RE.match(line)
        if hexed:
            frames.append(hexed.group("dso"))
    return out


def attribute(stacks: list[tuple[list[str], int]], resolve: Callable[[str], str | None]) -> dict:
    """Shares of the samples per workspace file (the deepest workspace frame of each
    stack) and, for stacks with none, per outside package or library."""
    files: Counter = Counter()
    outside: Counter = Counter()
    total = 0
    for frames, count in stacks:
        total += count
        home = next((r for r in (resolve(f) for f in reversed(frames)) if r), None)
        if home:
            files[home] += count
        else:
            outside[_outside(frames[-1] if frames else "")] += count
    if not total:
        return {"samples": 0, "files": {}, "outside": {}}
    return {"samples": total,
            "files": {f: round(n / total, 4) for f, n in files.most_common()},
            "outside": {k: round(n / total, 4) for k, n in outside.most_common(12)}}


def cprofile_files(path: str, resolve: Callable[[str], str | None]) -> dict:
    """A cProfile dump's self time per workspace file, the rest per outside package."""
    stats = pstats.Stats(path).stats  # type: ignore[attr-defined]
    files: Counter = Counter()
    outside: Counter = Counter()
    for (filename, _line, _func), (_cc, _nc, tottime, _ct, _callers) in stats.items():
        home = resolve(filename) if filename and filename[0] not in "<~" else None
        label = "built-ins" if filename == "~" else "python" if filename.startswith("<") else _outside(filename)
        (files if home else outside)[home or label] += tottime
    total = sum(files.values()) + sum(outside.values())
    if not total:
        return {"samples": 0, "files": {}, "outside": {}}
    return {"samples": len(stats),
            "files": {f: round(t / total, 4) for f, t in files.most_common() if t / total >= 5e-5},
            "outside": {k: round(t / total, 4) for k, t in outside.most_common(12) if t / total >= 5e-5},
            "note": "cProfile: self time per file, without the native code it calls"}


# ── taking one ───────────────────────────────────────────────────────────────

def profile(argv: list[str], out_dir: str, root: str, files: list[str], *,
            timeout: float | None = None, case: str = "") -> dict:
    """Run *argv* once under the profiler that fits, attribute its time to the workspace
    files, write ``profile.json`` in *out_dir* and return it; never raises."""
    tool, why = choose(argv)
    result: dict = {"tool": tool, "case": case}
    if not tool:
        result["unavailable"] = why
        return _write(out_dir, result)
    resolve = resolver(root, os.getcwd(), files)
    log = os.path.join(out_dir, "profile.log")
    try:
        with open(log, "w") as fh:
            if tool == "py-spy":
                raw = os.path.join(out_dir, "profile.raw")
                cmd = ["py-spy", "record", "--native", "--format", "raw", "--rate", "250", "-o", raw, "--", *argv]
                proc_run.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
                if not os.path.isfile(raw):      # --native is not supported everywhere
                    proc_run.run([c for c in cmd if c != "--native"], stdout=fh,
                                 stderr=subprocess.STDOUT, timeout=timeout)
                with open(raw, encoding="utf-8", errors="replace") as rf:
                    result.update(attribute(pyspy_stacks(rf.read()), resolve))
            elif tool == "cprofile":
                dump = os.path.join(out_dir, "profile.pstats")
                proc_run.run([argv[0], "-m", "cProfile", "-o", dump, *argv[1:]], stdout=fh,
                             stderr=subprocess.STDOUT, timeout=timeout)
                result.update(cprofile_files(dump, resolve))
            else:
                data = os.path.join(out_dir, "perf.data")
                proc_run.run(["perf", "record", "-g", "-F", "499", "-o", data, "--", *argv],
                             stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
                script = proc_run.run(["perf", "script", "-i", data, "-F", "ip,sym,dso,srcline"],
                                      stdout=subprocess.PIPE, stderr=fh, text=True, timeout=timeout)
                result.update(attribute(perf_stacks(script.stdout or ""), resolve))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["unavailable"] = f"{tool} failed: {exc}"
        return _write(out_dir, result)
    if not result.get("samples"):
        result["unavailable"] = f"{tool} recorded nothing (see profile.log)"
    if any(re.search(r"cuda|hip|nvrtc|rocm", k, re.I) for k in result.get("outside", {})):
        result["note"] = "GPU time is not seen: the profile shows the run's CPU side"
    return _write(out_dir, result)


def _write(out_dir: str, result: dict) -> dict:
    try:
        write_json(os.path.join(out_dir, PROFILE_FILE), result, indent=2)
    except OSError:
        pass
    return result


def top(profile_: dict, n: int = 5) -> dict:
    """The profile in a few lines for the agent: the *n* costliest workspace files and the
    share spent outside the workspace."""
    if not profile_ or profile_.get("unavailable"):
        return {"unavailable": (profile_ or {}).get("unavailable", "no profile")}
    files = list((profile_.get("files") or {}).items())[:n]
    return {"tool": profile_.get("tool"), "files": dict(files),
            "outside": round(sum((profile_.get("outside") or {}).values()), 4),
            **({"note": profile_["note"]} if profile_.get("note") else {})}
