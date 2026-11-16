"""callgrind, where perf may not run: valgrind counts the instructions each function and
line executes on a simulated CPU, with no privilege; many times slower, and instructions
are not time. A library function's count goes to the workspace functions that call it,
through library calls too, by what each call cost; a function's total is its own plus
its calls'. A marked region is recorded alone (``callgrind_control -i``).
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from typing import Callable, TextIO

from .. import proc_run
from .frames import Frame, fortran_name

_NAMED_RE = re.compile(r"^\(\d+\)\s*")


def _name(value: str) -> str:
    """A name with any ``(id)`` compression left over removed."""
    return _NAMED_RE.sub("", value.strip())


def read(text: str) -> tuple[Counter, dict, Counter]:
    """A callgrind file (``--compress-strings=no --compress-pos=no``): each function's own
    instructions per (file, function, line), the calls between functions with their
    inclusive cost, and each function's file or library."""
    own: Counter = Counter()
    calls: dict[tuple[str, str], int] = defaultdict(int)
    home: dict[str, str] = {}
    ob = fl = fi = fn = cfn = ""
    in_call = False
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        key, eq, value = line.partition("=")
        if eq and key in ("ob", "fl", "fi", "fe", "fn", "cfn", "cob", "cfi", "cfl", "calls"):
            if key == "ob":
                ob = _name(value)
            elif key == "fl":
                fl, fi = _name(value), ""
            elif key in ("fi", "fe"):
                fi = _name(value)
            elif key == "fn":
                fn, fi = _name(value), ""
                where = fl if fl and fl != "???" else ob
                home.setdefault(fn, where)
            elif key == "cfn":
                cfn = _name(value)
            elif key == "calls":
                in_call = True
            continue
        parts = line.split()
        if not parts or not (parts[0].isdigit() or parts[0][0] in "+-*"):
            continue
        cost = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        if in_call:
            if fn and cfn:
                calls[(fn, cfn)] += cost
            in_call = False
            continue
        where = fi or fl
        if where in ("", "???"):
            where = ob
        own[(where, fn, int(parts[0]) if parts[0].isdigit() else 0)] += cost
    return own, dict(calls), Counter(home)


def stacks(own: Counter, calls: dict, home: dict, resolve: Callable[[str], str | None]
           ) -> tuple[list[tuple[list[Frame], int]], dict[str, int]]:
    """Two-frame stacks for ``cpu.attribute``: a workspace function's own count on itself,
    a library function's shared among the workspace functions that call it, through any
    chain of library calls, by what each call cost; and each function's inclusive count."""
    callers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    inclusive: Counter = Counter()
    for (src, dst), cost in calls.items():
        if src != dst:
            callers[dst].append((src, cost))
            inclusive[src] += cost
    for (_where, fn, _line), cost in own.items():
        inclusive[fn] += cost
    by_symbol = getattr(resolve, "symbol", None) or (lambda *_: None)

    def mine(fn: str, where: str) -> bool:
        return bool(resolve(where) or by_symbol(fortran_name(fn, where), where))
    memo: dict[str, dict[str, float]] = {}

    def owners(fn: str, seen: frozenset = frozenset()) -> dict[str, float]:
        """The workspace functions *fn*'s cost belongs to, with their shares."""
        if fn in memo:
            return memo[fn]
        if mine(fn, home.get(fn, "")):
            return {fn: 1.0}
        by = [(c, w) for c, w in callers.get(fn) or [] if c not in seen]
        weight = sum(w for _, w in by)
        found: dict[str, float] = defaultdict(float)
        if weight and len(seen) < 32:
            for caller, w in by:
                for owner, share in owners(caller, seen | {fn}).items():
                    found[owner] += share * w / weight
        if not seen:
            memo[fn] = dict(found)
        return found
    out: list[tuple[list[Frame], int]] = []
    for (where, fn, line), cost in own.items():
        frame = Frame(where, fortran_name(fn, where), line)
        owned = {} if mine(fn, where) else owners(fn)
        if not owned:
            out.append(([frame], cost))
            continue
        for owner, share in owned.items():
            src = home.get(owner, "")
            out.append(([Frame(src, fortran_name(owner, src)), frame], int(cost * share)))
    return out, {fortran_name(k, home.get(k, "")): v for k, v in inclusive.items()}


def sample(target, out_dir: str, resolve: Callable[[str], str | None], log: TextIO,
           timeout: float | None, regions: bool = False) -> dict:
    """callgrind's profile of *target*, its measured region alone with *regions*."""
    from .cpu import attribute
    for old in glob.glob(os.path.join(out_dir, "callgrind.out.*")):
        os.unlink(old)
    out = os.path.join(out_dir, "callgrind.out.%p")
    windowed = regions and bool(shutil.which("callgrind_control"))
    tool = ["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}", "--compress-strings=no",
            "--compress-pos=no", "--trace-children=yes", *(["--instr-atstart=no"] if windowed else [])]

    def toggle(word: str, pids: list[str]) -> None:
        for pid in pids:
            try:
                subprocess.run(["callgrind_control", "-i", "on" if word == "begin" else "off", pid],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
    proc_run.run(target.wrap(tool, out_dir), stdout=log, stderr=subprocess.STDOUT,
                 timeout=timeout, region=toggle if windowed else None)
    own: Counter = Counter()
    calls: dict = defaultdict(int)
    home: dict = {}
    for path in glob.glob(os.path.join(out_dir, "callgrind.out.*")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            o, c, h = read(fh.read())
        own.update(o)
        for k, v in c.items():
            calls[k] += v
        for k, v in h.items():
            home.setdefault(k, v)
    found, inclusive = stacks(own, calls, home, resolve)
    result = attribute(found, resolve)
    total = result.get("samples") or 0
    left = total * (1 - result.get("harness", 0))
    for key, row in (result.get("functions") or {}).items():
        func = key.split("::", 1)[1] if "::" in key else key
        if left and inclusive.get(func):
            row["total"] = round(min(inclusive[func] / left, 1.0), 4)
    result["unit"] = "instructions"
    notes = ["callgrind counts instructions on a simulated CPU, not time: memory stalls do not show"]
    if windowed:
        result["window"] = "measured region"
    result["note"] = "; ".join(notes)
    return result
