"""What the machine code of the costliest kernel or function spends its instructions on:
the mix of opcodes weighted by how often each ran (a GPU kernel's SASS under Nsight
Compute's source counters) or was sampled (a CPU function under ``perf annotate``), the
calls into slow paths (an IEEE division, a libm routine), and spills to local memory.

A memory profile cannot see a kernel saturated by its own address arithmetic or by a
division routine called once per point; the instructions do.
"""
from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from typing import TextIO

from .. import proc_run
from . import gpu
from .target import Target

# Opcode classes, by the opcode's first word, tried in this order: an x86 ``addsd`` is a
# floating add before it is an ``add``, an ``LDL`` a spill before it is a load. GPU (SASS)
# and x86/Arm mnemonics side by side: the classes, not the spelling, are what a reader compares.
_CLASSES = (
    ("spill", re.compile(r"^(LDL|STL)$", re.I)),
    ("special", re.compile(r"^(MUFU|v?div[sp][sd]|v?sqrt[sp][sd]|v?rcp|v?rsqrt|fdiv|fsqrt)", re.I)),
    ("floating", re.compile(r"^(FFMA|FADD|FMUL|FMNMX|FSETP|DFMA|DADD|DMUL|HFMA2|HADD2|HMUL2|"
                            r"v?fn?m(add|sub)\w*|v?(add|sub|mul|max|min)[sp][sd]|fadd|fmul|fmla|fmadd)$", re.I)),
    ("call", re.compile(r"^(CALL|call|callq|bl|blr)$", re.I)),
    ("load", re.compile(r"^(LDG|LDS|LDC|LD|ULDC|v?mov[ua]p[sd]|v?movs[sd]|v?broadcast\w*|ldr|ldp|ld1)$", re.I)),
    ("store", re.compile(r"^(STG|STS|ST|v?movntp[sd]|str|stp|st1)$", re.I)),
    ("branch", re.compile(r"^(BRA|BSYNC|BSSY|EXIT|RET|ISETP|j[a-z]+|cmp|test|b|b\.\w+|cbn?z|ret)$", re.I)),
    ("integer", re.compile(r"^(IMAD|IADD3?|LEA|SHF|SHL|SHR|ISCADD|LOP3|IABS|IMNMX|"
                           r"lea|add|sub|imul|shl|shr|sar|inc|dec|and|or|xor|neg|madd|lsl|lsr)$", re.I)),
)
_TOP = 10


def classify(opcode: str) -> str:
    """The class of *opcode* (its modifiers dropped), "other" when none fits."""
    # SASS modifiers follow a dot (IMAD.WIDE.U32); an Arm condition does too (b.ne), and an
    # AT&T mnemonic may carry its operand size (addq, leal, callq).
    base = opcode.split(".")[0] if opcode.isupper() else opcode
    tries = [base] + ([base[:-1]] if base.islower() and base[-1:] in "lqwb" else [])
    for word in tries:
        for name, pattern in _CLASSES:
            if pattern.match(word):
                return name
    return "other"


def mix(lines: list[tuple[str, float]]) -> dict:
    """The mix of ``(instruction text, weight)``: each class's share, the costliest opcodes,
    the calls (by target) and the spills. Weights are executions or samples."""
    by_op: Counter = Counter()
    by_class: Counter = Counter()
    calls: Counter = Counter()
    total = 0.0
    for text, weight in lines:
        words = re.sub(r"^@!?U?P\w+\s+", "", text.strip()).split()
        if not words or weight <= 0:
            continue
        op = words[0].rstrip(";")
        kind = classify(op)
        total += weight
        by_op[op] += weight
        by_class[kind] += weight
        if kind == "call":
            target = re.search(r"[`(<]+([^`)>]+)", text)
            calls[(target.group(1) if target else " ".join(words[1:2])).strip()] += weight
    if not total:
        return {}
    share = lambda c: {k: round(v / total, 4) for k, v in c.most_common(_TOP)}  # noqa: E731
    found = {"executed": round(total), "classes": share(by_class), "opcodes": share(by_op)}
    if calls:
        found["calls"] = share(calls)
    if by_class.get("spill"):
        found["spills"] = round(by_class["spill"] / total, 4)
    return found


def notes(found: dict) -> list[str]:
    """What the mix says, in the words a next edit is chosen in."""
    classes, said = found.get("classes") or {}, []
    if classes.get("integer", 0) > max(classes.get("floating", 0), 0.25):
        said.append("integer arithmetic outweighs the floating point: index and address computation; "
                    "a base pointer with constant offsets, hoisted out of the inner loop, removes most of it")
    if found.get("calls"):
        said.append("calls into " + ", ".join(list(found["calls"])[:3])
                    + ": a slow path run per point (an IEEE division, a libm routine); a reciprocal "
                      "computed once, or the constant folded in, avoids it")
    if classes.get("special", 0) > 0.05:
        said.append("divisions and square roots take a visible share: fold the constants they divide by")
    if found.get("spills"):
        said.append("registers spill to local memory: fewer live values, or a smaller unroll")
    return said


# ── GPU: SASS under Nsight Compute ──────────────────────────────────────────

def ncu_source(csv_path: str) -> list[tuple[str, float]]:
    """``ncu --page source --csv --print-source sass``: each instruction and how often it ran."""
    rows = gpu.csv_rows(csv_path)
    if not rows:
        raise ValueError("ncu exported no source")
    src = gpu.column(rows, r"Source", r"SASS")
    count = gpu.column(rows, r"Instructions Executed", r"inst_executed", r"Warp Stall Sampling.*")
    return [(r[src], gpu.number(r[count])) for r in rows if r.get(src)]


def sass(target: Target, out_dir: str, name: str, log: TextIO, timeout: float | None,
         regions: bool = False) -> dict:
    """The kernel *name*'s instruction mix, from its first launch (in the marked region with
    *regions*)."""
    gpu_dir = os.path.join(out_dir, "gpu")
    os.makedirs(gpu_dir, exist_ok=True)
    report, out = os.path.join(gpu_dir, "asm"), os.path.join(gpu_dir, "asm.csv")
    regex = re.escape(gpu.kernel_base(name) or name)
    for path in (report + ".ncu-rep", out):
        if os.path.exists(path):
            os.unlink(path)
    proc_run.run(target.wrap(["ncu", "--section", "SourceCounters", "--kernel-name", f"regex:{regex}",
                              "--launch-count", "1", "-o", report,
                              *(["--profile-from-start", "off"] if regions else []), "--"], out_dir),
                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                 region=gpu.marked if regions else None)
    if not os.path.isfile(report + ".ncu-rep"):
        raise ValueError(f"ncu profiled no launch of {name}")
    with open(out, "w") as fh:
        proc_run.run(["ncu", "-i", report + ".ncu-rep", "--page", "source", "--csv", "--print-source", "sass"],
                     stdout=fh, stderr=log, timeout=timeout)
    found = mix(ncu_source(out))
    if not found:
        raise ValueError(f"ncu counted no instruction of {name}")
    return {"tool": "ncu", "kernel": name, **found, "notes": notes(found)}


# ── CPU: perf annotate ──────────────────────────────────────────────────────

# `  12.34 :   401a2b:  vfmadd231pd %ymm1,%ymm2,%ymm0` — a share of the samples, an address,
# the instruction.
_ANNOTATED = re.compile(r"^\s*([\d.]+)\s*:\s*[0-9a-f]+:\s*(.+)$")


def annotated(text: str) -> list[tuple[str, float]]:
    """``perf annotate --stdio``: each instruction and its share of the function's samples."""
    out = []
    for line in text.splitlines():
        m = _ANNOTATED.match(line)
        if m:
            out.append((m.group(2), float(m.group(1))))
    return out


def annotate(target: Target, out_dir: str, function: str, log: TextIO, timeout: float | None) -> dict:
    """The function *function*'s instruction mix, sampled over one run of *target*."""
    data = os.path.join(out_dir, "asm.perf.data")
    if os.path.exists(data):
        os.unlink(data)
    from .cpu import PERF_HZ
    proc_run.run(target.wrap(["perf", "record", "-F", str(PERF_HZ), "-o", data, "--"], out_dir),
                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if not os.path.isfile(data):
        raise ValueError("perf recorded nothing")
    shown = proc_run.run(["perf", "annotate", "--stdio", "-i", data, "--symbol", function],
                         stdout=subprocess.PIPE, stderr=log, text=True, timeout=timeout)
    found = mix(annotated(shown.stdout or ""))
    if not found:
        raise ValueError(f"perf annotate found no sample in {function} (a C++ name may need its mangled form)")
    found.pop("executed", None)  # shares of the function's samples, not counts
    return {"tool": "perf annotate", "function": function, **found, "notes": notes(found)}
