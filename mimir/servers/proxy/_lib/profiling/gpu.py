"""The GPU side of a profile: which vendor's GPU a run uses, its kernels and copies traced
by Nsight Systems (NVIDIA) or rocprofv3 (AMD) as shares of the run's wall time, each
kernel given to the workspace file that defines it; and, on request, one kernel's
hardware counters by Nsight Compute or rocprofv3.

The tools' CSV columns move between versions, so they are found by header name; a
header that is not there says so rather than reading the wrong column.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from typing import Callable, TextIO

from .....shared import code_map
from .. import proc_run
from .cpu import is_python
from .frames import FORTRAN_EXT, fortran_name, is_fortran

KERNELS_KEPT = 20
_UNSET = (None, "", "-1", "NoDevFiles")
_AMD_RE = re.compile(r"amdhip|libhip|rocm|hsa-runtime|rocblas|miopen", re.I)
_NVIDIA_RE = re.compile(r"cudart|libcuda\b|nvrtc|cublas|cudnn|cufft|cusparse|nccl|_cuda", re.I)
_GPU_EXT = (".c", ".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx")
_GLOBAL_RE = re.compile(r"__global__\b")
_FORTRAN_GLOBAL_RE = re.compile(r"attributes\s*\(\s*global\s*\)\s*subroutine\s+(?P<name>\w+)", re.I)
# A C or C++ function's definition: a name, its parameters, then its body.
_C_DEF_RE = re.compile(r"^[\w:<>,\*&\s]*?\b(\w+)\s*\([^;{}]*\)\s*(?:const\s*)?\{", re.M)
# Kernels a compiler makes of a loop: OpenACC (NVHPC), OpenMP (NVHPC), OpenMP offload
# (Clang, AMD flang, Cray), each naming the procedure and the loop's line.
_GENERATED = (
    re.compile(r"^(?P<proc>\w+?)_(?P<line>\d+)_gpu(?:__\w+)?$"),
    re.compile(r"^nvkernel_(?P<proc>\w+?)_F\d+L(?P<line>\d+)_\d+$"),
    re.compile(r"^__omp_offloading_[0-9a-f]+_[0-9a-f]+_(?P<proc>\w+?)_l(?P<line>\d+)(?:_\w+)?$"),
)


# ── which GPU, which tool ────────────────────────────────────────────────────

def _linked(argv: list[str]) -> str:
    """The shared libraries a compiled *argv* links, as ldd prints them; "" otherwise."""
    if not argv or is_python(argv):
        return ""
    exe = shutil.which(argv[0]) or argv[0]
    if not os.path.isfile(exe) or not shutil.which("ldd"):
        return ""
    try:
        return subprocess.run(["ldd", exe], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_vendor(argv: list[str], seen_outside: list[str] | tuple[str, ...] = ()) -> str:
    """"nvidia" or "amd" when the run uses a GPU, else "": from the libraries it links,
    those its CPU profile ran into, then the devices the job was given."""
    hints = " ".join([_linked(argv), *seen_outside])
    if _AMD_RE.search(hints):
        return "amd"
    if _NVIDIA_RE.search(hints):
        return "nvidia"
    if any(os.environ.get(v) not in _UNSET for v in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")):
        return "amd"
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in _UNSET:
        return "amd" if shutil.which("rocm-smi") and not shutil.which("nvidia-smi") else "nvidia"
    return ""


def choose_gpu(vendor: str, detail: bool = False) -> tuple[str, str]:
    """The GPU tracer for *vendor* (its kernel profiler when *detail*), or "" with why."""
    tool = {("nvidia", False): "nsys", ("nvidia", True): "ncu",
            ("amd", False): "rocprofv3", ("amd", True): "rocprofv3"}.get((vendor, detail), "")
    if not tool:
        return "", "the run uses no GPU this profiler knows"
    if shutil.which(tool):
        return tool, ""
    suite = {"nsys": "Nsight Systems", "ncu": "Nsight Compute", "rocprofv3": "ROCm"}[tool]
    return "", f"{tool} ({suite}) is not on PATH; load the module that provides it in the job's environment"


# ── reading what a tracer wrote ──────────────────────────────────────────────

def _rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.startswith(("==", "#"))]
    return list(csv.DictReader(lines))


def _column(rows: list[dict[str, str]], *patterns: str) -> str:
    """The first header matching one of *patterns*, or ValueError naming them."""
    headers = list(rows[0]) if rows else []
    for pattern in patterns:
        hit = next((h for h in headers if h and re.fullmatch(pattern, h.strip(), re.I)), None)
        if hit:
            return hit
    raise ValueError(f"no column {' or '.join(patterns)} in {headers}")


def _number(text: str) -> float:
    return float((text or "0").replace(",", "").strip() or 0)


def _direction(text: str) -> str:
    t = re.sub(r"[\s_-]", "", text.lower())
    if "memset" in t:
        return "memset"
    for key, words in (("HtoD", ("hosttodevice", "htod")), ("DtoH", ("devicetohost", "dtoh")),
                       ("DtoD", ("devicetodevice", "dtod")), ("PtoP", ("peertopeer", "ptop"))):
        if any(w in t for w in words):
            return key
    return "other"


def _kernel_table(rows: list[dict[str, str]]) -> dict[str, tuple[float, int]]:
    name = _column(rows, r"Name", r"Kernel[_ ]?Name")
    ns = _column(rows, r"Total ?Time \(ns\)", r"TotalDurationNs", r"Total Time")
    calls = _column(rows, r"Instances", r"Calls", r"Count")
    table: dict[str, tuple[float, int]] = {}
    for r in rows:
        t, c = table.get(r[name], (0.0, 0))
        table[r[name]] = (t + _number(r[ns]), c + int(_number(r[calls])))
    return table


def _copy_table(rows: list[dict[str, str]]) -> dict[str, float]:
    op = _column(rows, r"Operation", r"Name")
    ns = _column(rows, r"Total ?Time \(ns\)", r"TotalDurationNs", r"Total Time")
    out: Counter = Counter()
    for r in rows:
        out[_direction(r[op])] += _number(r[ns])
    return dict(out)


# The unit a size column is in, by its header; nsys's older reports give KiB unlabelled.
_SIZE_UNITS = {"b": 1, "bytes": 1, "kb": 1e3, "kib": 2**10, "mb": 1e6, "mib": 2**20, "gb": 1e9, "gib": 2**30}


def _size_table(rows: list[dict[str, str]]) -> dict[str, float]:
    """Bytes copied per direction, from a column of sizes in the unit its header names."""
    op = _column(rows, r"Operation", r"Direction", r"Name", r"Kind")
    size = _column(rows, r"Total(?: \(\w+\))?", r"Bytes", r"Size(?: \(\w+\))?")
    unit = re.search(r"\((\w+)\)", size)
    scale = _SIZE_UNITS.get(unit.group(1).lower(), 1) if unit else (2**10 if size == "Total" else 1)
    out: Counter = Counter()
    for r in rows:
        out[_direction(r[op])] += _number(r[size]) * scale
    return {k: v for k, v in out.items() if k != "memset"}


def nsys_summary(prefix: str) -> dict:
    """``nsys stats`` CSVs written under *prefix*: kernels (ns, calls), copies (ns) and
    the bytes they moved."""
    kern = _first_file(prefix, ("cuda_gpu_kern_sum", "gpukernsum"))
    mem = _first_file(prefix, ("cuda_gpu_mem_time_sum", "gpumemtimesum"))
    size = _first_file(prefix, ("cuda_gpu_mem_size_sum", "gpumemsizesum"))
    if not kern:
        raise ValueError("nsys wrote no kernel summary (did the run launch any CUDA kernel?)")
    return {"kernels": _kernel_table(_rows(kern)), "memcpy": _copy_table(_rows(mem)) if mem and _rows(mem) else {},
            "memcpy_bytes": _size_table(_rows(size)) if size and _rows(size) else {}}


def rocprof_summary(out_dir: str) -> dict:
    """rocprofv3 ``--stats`` CSVs anywhere under *out_dir* (one set per process), summed."""
    kernels: dict[str, tuple[float, int]] = {}
    copies: Counter = Counter()
    found = False
    for path in glob.glob(os.path.join(out_dir, "**", "*kernel_stats.csv"), recursive=True):
        found = True
        for k, (t, c) in _kernel_table(_rows(path)).items():
            t0, c0 = kernels.get(k, (0.0, 0))
            kernels[k] = (t0 + t, c0 + c)
    for path in glob.glob(os.path.join(out_dir, "**", "*memory_copy_stats.csv"), recursive=True):
        rows = _rows(path)
        if rows:
            copies.update(_copy_table(rows))
    moved: Counter = Counter()
    for path in glob.glob(os.path.join(out_dir, "**", "*memory_copy_trace.csv"), recursive=True):
        rows = _rows(path)
        try:
            moved.update(_size_table(rows) if rows else {})
        except ValueError:
            pass                        # a version without sizes: durations only
    if not found:
        raise ValueError("rocprofv3 wrote no kernel statistics (did the run launch any HIP kernel?)")
    return {"kernels": kernels, "memcpy": dict(copies), "memcpy_bytes": dict(moved)}


def _first_file(prefix: str, reports: tuple[str, ...]) -> str:
    for report in reports:
        hits = sorted(glob.glob(f"{prefix}*_{report}.csv"))
        if hits:
            return hits[0]
    return ""


# ── a kernel's home ──────────────────────────────────────────────────────────

def kernel_base(name: str) -> str:
    """A kernel's bare name: demangled, without return type, namespace, template
    arguments or parameters."""
    if name.startswith("_Z") and shutil.which("c++filt"):
        try:
            name = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5).stdout.strip() or name
        except (OSError, subprocess.SubprocessError):
            pass
    name = re.sub(r"\s*\[clone [^\]]*\]$", "", name.strip())
    out, depth = [], 0
    for ch in name:                      # drop <...> however nested
        depth += ch == "<"
        if depth == 0:
            out.append(ch)
        depth -= ch == ">" and depth > 0
    bare = "".join(out).split("(")[0].strip()
    return re.split(r"::|\s", bare)[-1] if bare else ""


def kernel_index(root: str, files: list[str]) -> dict[str, list[str]]:
    """Every kernel the workspace writes as one, by name, with its files: ``__global__``
    (CUDA, HIP) and ``attributes(global)`` (CUDA Fortran, by lower-case name)."""
    index: dict[str, list[str]] = {}
    for rel, text in _sources(root, files):
        fortran = is_fortran(rel)
        for m in (_FORTRAN_GLOBAL_RE if fortran else _GLOBAL_RE).finditer(text):
            if fortran:
                name = m.group("name").lower()
            else:
                head = re.split(r"[{;]", text[m.end():m.end() + 400], maxsplit=1)[0]
                fn = re.search(r"(\w+)\s*\(", re.sub(r"__launch_bounds__\s*\([^)]*\)", "", head))
                if not fn:
                    continue
                name = fn.group(1)
            if rel not in index.setdefault(name, []):
                index[name].append(rel)
    return index


def generated(name: str) -> tuple[str, int]:
    """The procedure and line a compiler-generated kernel (OpenACC, OpenMP offload) was
    made from, or ("", 0) for a kernel written as one."""
    for pattern in _GENERATED:
        m = pattern.match(name)
        if m:
            proc = m.group("proc")
            if proc.startswith("_Z"):
                proc = kernel_base(proc)
            return fortran_name(proc).split("::")[-1], int(m.group("line"))
    return "", 0


def kernel_homes(root: str, files: list[str]) -> Callable[[str], tuple[str, int]]:
    """A function giving a kernel's workspace file (and line, for a generated one), or
    ("", 0) when no one file defines it."""
    index = kernel_index(root, files)
    procs: dict[str, list[str]] = {}
    for rel, text in _sources(root, files):
        if is_fortran(rel):
            names = {n.lower() for kind, n in code_map.parse(rel, text).exports if kind == "fproc"}
        else:
            names = set(_C_DEF_RE.findall(text))
        for n in names:
            procs.setdefault(n, []).append(rel)

    def home(name: str) -> tuple[str, int]:
        base = kernel_base(name)
        written = index.get(base) or index.get(base.lower()) or []
        if len(written) == 1:
            return written[0], 0
        proc, line = generated(base or name)
        # A Fortran external procedure keeps the compiler's trailing underscore.
        where = next((procs[p] for p in (proc, proc.lower(), proc.rstrip("_").lower()) if p in procs), [])
        return (where[0], line) if len(where) == 1 else ("", 0)
    return home


def _sources(root: str, files: list[str]):
    """The workspace's C, C++, CUDA, HIP and Fortran files with their text."""
    for rel in files:
        if not rel.lower().endswith(_GPU_EXT + FORTRAN_EXT):
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                yield rel, fh.read()
        except OSError:
            continue


def summarize(summary: dict, wall_s: float, home: Callable[[str], tuple[str, int]]) -> dict:
    """Kernels and copies as shares of *wall_s*, each kernel with its workspace file (see
    ``kernel_homes``), and the time the GPU did neither."""
    wall_ns = max(wall_s, 1e-9) * 1e9
    kernels = sorted(summary["kernels"].items(), key=lambda kv: -kv[1][0])
    busy_ns = sum(t for _, (t, _c) in kernels)
    copies = {k: round(t / wall_ns, 4) for k, t in summary.get("memcpy", {}).items() if t}
    files: Counter = Counter()
    shown: dict[str, dict] = {}
    for name, (t, calls) in kernels:
        where, line = home(name)
        if where:
            files[where] += t
        if len(shown) < KERNELS_KEPT:
            shown[name] = {"share": round(t / wall_ns, 4), "calls": calls,
                           **({"file": where} if where else {}), **({"line": line} if line else {})}
    busy = busy_ns / wall_ns
    result = {"wall_s": round(wall_s, 4), "busy": round(min(busy, 1.0), 4), "memcpy": copies,
              "idle": round(max(0.0, 1.0 - min(busy, 1.0) - sum(copies.values())), 4),
              "files": {f: round(t / wall_ns, 4) for f, t in files.most_common()},
              "kernels": shown,
              "outside": round((busy_ns - sum(files.values())) / wall_ns, 4)}
    if busy > 1.0:
        result["note"] = "kernels overlap (several streams or GPUs): their summed time exceeds the run's"
    return result


# ── taking one ───────────────────────────────────────────────────────────────

def trace(tool: str, argv: list[str], out_dir: str, root: str, files: list[str],
          log: TextIO, timeout: float | None, wall_s: float | None) -> dict:
    """Run *argv* once under the GPU tracer *tool* and summarize its kernels and copies
    against *wall_s*, the run's measured time (else the traced run's own)."""
    gpu_dir = os.path.join(out_dir, "gpu")
    os.makedirs(gpu_dir, exist_ok=True)
    device = DeviceMemory("nvidia" if tool == "nsys" else "amd")
    start = time.monotonic()
    if tool == "nsys":
        rep = os.path.join(gpu_dir, "trace")
        with device:
            proc_run.run(["nsys", "profile", "-t", "cuda,osrt", "-s", "none", "--force-overwrite", "true",
                          "-o", rep, "--", *argv], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        spent = time.monotonic() - start
        report = next((p for p in (rep + ".nsys-rep", rep + ".qdrep") if os.path.isfile(p)), "")
        if not report:
            raise ValueError("nsys wrote no report (see profile.log)")
        proc_run.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum",
                      "--format", "csv", "--force-export", "true", "--output", rep, report],
                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        summary = nsys_summary(rep)
    else:
        with device:
            proc_run.run(["rocprofv3", "--kernel-trace", "--memory-copy-trace", "--stats",
                          "--output-format", "csv", "-d", gpu_dir, "--", *argv],
                         stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        spent = time.monotonic() - start
        summary = rocprof_summary(gpu_dir)
    result = summarize(summary, wall_s or spent, kernel_homes(root, files))
    if summary.get("memcpy_bytes"):
        result["memcpy_mb"] = {k: round(v / 2**20, 1) for k, v in summary["memcpy_bytes"].items()}
    if device.peak_mb is not None:
        result["peak_device_mb"] = round(device.peak_mb, 1)
        if device.note:
            result["note"] = "; ".join(filter(None, [result.get("note"), device.note]))
    if not wall_s:
        result["note"] = (result.get("note", "") + "; " if result.get("note") else "") + \
            "shares of the traced run's wall time, the tracer's start-up included"
    return result


class DeviceMemory:
    """The GPU memory a run held at its highest, sampled while it runs: NVIDIA's per
    process (those that came and went with the run), AMD's for the whole device above
    what it held before."""

    INTERVAL_S = 0.2

    def __init__(self, vendor: str) -> None:
        self.vendor, self.samples, self.note = vendor, [], ""
        self.peak_mb: float | None = None
        self._stop = threading.Event()

    def __enter__(self) -> "DeviceMemory":
        self.before = sample_device(self.vendor)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def _watch(self) -> None:
        while not self._stop.wait(self.INTERVAL_S):
            found = sample_device(self.vendor)
            if found is not None:
                self.samples.append(found)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(5)
        after = sample_device(self.vendor)
        if self.before is None or not self.samples:
            return
        if self.vendor == "amd":
            self.peak_mb = max(0.0, max(s.get("device", 0.0) for s in self.samples) - self.before.get("device", 0.0))
            self.note = "peak device memory is the whole GPU's, above what it held before the run"
            return
        # The run's processes are the ones that were not there before it and are gone after.
        others = set(self.before) | set(after or {})
        self.peak_mb = max(sum(mb for pid, mb in s.items() if pid not in others) for s in self.samples)


def sample_device(vendor: str) -> dict[str, float] | None:
    """GPU memory in use now, in MB: per process for NVIDIA, per device for AMD."""
    try:
        if vendor == "nvidia":
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            used: Counter = Counter()
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


_NCU_METRICS = {"sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active"}
_ROCPROF_COUNTERS = {"sm_pct": "VALUBusy", "dram_pct": "MemUnitBusy"}


def ncu_detail(csv_path: str, name: str) -> dict:
    """``ncu --csv --page raw`` output: the first profiled launch of the kernel."""
    rows = [r for r in _rows(csv_path) if (r.get("ID") or "").strip().isdigit()]
    if not rows:
        raise ValueError(f"ncu profiled no launch of {name}")
    row = rows[0]
    found = {k: round(_number(row[m]), 2) for k, m in _NCU_METRICS.items() if row.get(m) not in (None, "")}
    return {"name": row.get("Kernel Name", name), **found}


def rocprof_detail(out_dir: str, name: str) -> dict:
    """rocprofv3 ``--pmc`` counter CSVs: each counter's mean over the kernel's dispatches."""
    sums: Counter = Counter()
    counts: Counter = Counter()
    kernel = name
    for path in glob.glob(os.path.join(out_dir, "**", "*counter_collection.csv"), recursive=True):
        rows = _rows(path)
        if not rows:
            continue
        kcol, ccol, vcol = (_column(rows, r"Kernel[_ ]?Name"), _column(rows, r"Counter[_ ]?Name"),
                            _column(rows, r"Counter[_ ]?Value"))
        for r in rows:
            kernel = r[kcol]
            sums[r[ccol]] += _number(r[vcol])
            counts[r[ccol]] += 1
    if not counts:
        raise ValueError(f"rocprofv3 counted nothing for {name}")
    return {"name": kernel, **{k: round(sums[c] / counts[c], 2) for k, c in _ROCPROF_COUNTERS.items() if counts[c]}}


def bound(detail: dict) -> str:
    """What limits a kernel, from its compute and memory throughput (% of peak)."""
    sm, dram = detail.get("sm_pct"), detail.get("dram_pct")
    if sm is None or dram is None:
        return "unknown"
    if max(sm, dram) < 40:
        return "latency"
    return "memory" if dram > sm else "compute"


def kernel_detail(tool: str, argv: list[str], out_dir: str, name: str, log: TextIO,
                  timeout: float | None) -> dict:
    """One launch of the kernel *name* under its vendor's kernel profiler."""
    gpu_dir = os.path.join(out_dir, "gpu")
    os.makedirs(gpu_dir, exist_ok=True)
    regex = re.escape(kernel_base(name) or name)
    if tool == "ncu":
        out = os.path.join(gpu_dir, "ncu.csv")
        proc_run.run(["ncu", "--csv", "--page", "raw", "--log-file", out, "--kernel-name", f"regex:{regex}",
                      "--launch-count", "1", "--metrics", ",".join(_NCU_METRICS.values()), "--", *argv],
                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        found = ncu_detail(out, name)
    else:
        proc_run.run(["rocprofv3", "--kernel-include-regex", regex, "--pmc", *_ROCPROF_COUNTERS.values(),
                      "--output-format", "csv", "-d", gpu_dir, "--", *argv],
                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        found = rocprof_detail(gpu_dir, name)
    return {"tool": tool, **found, "bound": bound(found)}
