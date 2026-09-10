"""
MCP Platform Server
===================
Reports what this host actually is — CPU/SIMD, memory, GPU, Slurm, modules,
toolchains, Python environments. Facts only: the architecture-aware *advice* that
used to live here was a frozen lookup table the model already knows better, paid for
with a full hardware probe per call.

Every *live* fact is probed on demand and never cached to disk, so it cannot go
stale; the collectors whose answer cannot change mid-process are memoized. The one
persisted artefact is the **module catalogue**. A site's module tree runs to hundreds
or thousands of entries — far more than fits in a context window — so it is indexed
once under ``<STATE_DIR>/platform/modules/`` and queried with ``platform_search``
instead of being dumped in truncated slices. Alongside it rides a small **digest** of
the stable, high-impact facts that *do* fit: host architecture, the kinds of compute
node the cluster has, its partitions, and the toolchains present.

The catalogue is rebuilt when — and only when — a deterministic fingerprint of the
tree changes (hostname, effective MODULEPATH, a stat-hash of the modulefile trees and
of Lmod's own spider cache). No TTL is ever consulted.
"""

import hashlib
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import json
from datetime import datetime, timezone
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from module_env import module_probe_script
from capabilities import tool_caps, CACHEABLE, ENV_DISCOVERY
from responses import ok
from slurm_nodes import aggregate_node_types, parse_scontrol_nodes, stable_signature, stable_types
from state_paths import state_dir
import embed as _embed
import vector_cache

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


mcp = FastMCP(
    "PlatformServer",
    debug=False,
    log_level="ERROR",
)


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: list[str], timeout: int = 8) -> dict:
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(e)}


def _run_shell(script: str, timeout: int = 10) -> dict:
    return _run(["bash", "-lc", script], timeout=timeout)


def _parse_lscpu(text: str) -> dict:
    info = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        # Normalize key to title case so lookups are distro-independent.
        info[k.strip().lower()] = v.strip()
    return info


# ISA extensions worth reporting, per architecture. There is no portable name for
# "the vector unit": asking an aarch64 host whether it has AVX-512 always answers no,
# which reads as "no SIMD" rather than "a different SIMD". lscpu prints these under
# "Flags:" on x86 and "Features:" on aarch64, so both keys are read.
_ISA_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "x86_64":  ("avx2", "avx512f", "avx512bw", "avx512vl", "fma", "amx_tile"),
    "aarch64": ("asimd", "sve", "sve2", "bf16", "i8mm"),
    "ppc64le": ("vsx",),
}


@lru_cache(maxsize=1)
def _collect_cpu() -> dict:
    data = {
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if _cmd_exists("lscpu"):
        out = _run(["lscpu"])
        if out["ok"]:
            ls = _parse_lscpu(out["stdout"])
            flags = set((ls.get("flags") or ls.get("features") or "").split())
            known = _ISA_EXTENSIONS.get(data["arch"], ())
            data.update(
                {
                    "model": ls.get("model name", ""),
                    "sockets": ls.get("socket(s)", ""),
                    "cores_per_socket": ls.get("core(s) per socket", ""),
                    "threads_per_core": ls.get("thread(s) per core", ""),
                    "numa_nodes": ls.get("numa node(s)", ""),
                    "simd": {name: name in flags for name in known},
                }
            )
            if not known:
                data["simd_note"] = f"No ISA extension list known for {data['arch']}."
    return data


def _collect_memory() -> dict:
    data = {}
    if _cmd_exists("free"):
        out = _run(["free", "-b"])
        if out["ok"]:
            for line in out["stdout"].splitlines():
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 4:
                        total = int(parts[1])
                        used = int(parts[2])
                        free = int(parts[3])
                        data = {
                            "total_gb": round(total / 1e9, 2),
                            "used_gb": round(used / 1e9, 2),
                            "free_gb": round(free / 1e9, 2),
                        }
                    break
    return data


# Vendor CLI -> the tool that proves a GPU of that vendor is present locally. Only the
# NVIDIA output is parsed into devices; the others are detected and reported as such,
# because claiming "no GPU" on a machine whose accelerator this probe cannot read is
# worse than saying so. Cluster-wide GPU truth comes from Slurm GRES (slurm_nodes),
# which is vendor-neutral.
_GPU_PROBES = (("nvidia", "nvidia-smi"), ("amd", "rocm-smi"), ("intel", "xpu-smi"))

# compute_cap is queried alongside the descriptive fields because it is the one that
# decides a build: on a CUDA/Kokkos project the arch flag comes from it, and nothing
# else here substitutes. Reported without it, a host is described by its marketing
# name, and the gap gets filled by inference from that name — which is exactly where
# it breaks (B200 is sm_100, B300 is sm_103), silently, until the first kernel launch.
_GPU_FIELDS_BASE = ("name", "memory", "driver")
_GPU_FIELDS = _GPU_FIELDS_BASE + ("compute_cap",)
_GPU_QUERY_BASE = ("nvidia-smi --query-gpu=name,memory.total,driver_version"
                   " --format=csv,noheader")
_GPU_QUERY = ("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap"
              " --format=csv,noheader")


@lru_cache(maxsize=1)
def _collect_gpu() -> dict:
    present = [vendor for vendor, cmd in _GPU_PROBES if _cmd_exists(cmd)]
    if not present:
        return {"available": False, "probed": [cmd for _, cmd in _GPU_PROBES]}
    if "nvidia" not in present:
        return {
            "available": True, "vendors": present, "devices": [],
            "note": "Accelerator detected but not enumerated: only the NVIDIA probe is "
                    "parsed here. Ask Slurm (slurm_nodes) for GPU type and count.",
        }
    out = _run_shell(_GPU_QUERY)
    fields = _GPU_FIELDS
    if not out["ok"]:
        # An nvidia-smi too old to know a field rejects the whole query rather than
        # the field, which would cost us the enumeration entirely. Retry without the
        # newer field before reporting the host as unreadable.
        out = _run_shell(_GPU_QUERY_BASE)
        fields = _GPU_FIELDS_BASE
    if not out["ok"]:
        return {"available": False, "vendors": present, "error": out["stderr"].strip()}
    gpus = []
    for line in out["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_GPU_FIELDS_BASE):
            continue
        # nvidia-smi prints '[N/A]' / '[Not Supported]' for a field this driver cannot
        # answer. Dropping it says "unknown"; keeping it would read as a real value.
        gpu = {k: v for k, v in zip(fields, parts) if not v.startswith("[")}
        if gpu.get("name"):
            gpus.append(gpu)
    return {"available": bool(gpus), "vendors": present, "count": len(gpus), "devices": gpus}


@lru_cache(maxsize=1)
def _collect_slurm() -> dict:
    if not _cmd_exists("sinfo"):
        return {"available": False}
    out = _run(["sinfo", "--version"])
    if not out["ok"]:
        return {"available": False, "error": out["stderr"].strip()}
    version = out["stdout"].strip() or out["stderr"].strip()
    return {"available": True, "version": version}


def _collect_sinfo() -> dict:
    if not _cmd_exists("sinfo"):
        return {"available": False}
    out = _run(
        ["sinfo", "--format=%P %.5a %.10l %.6D %.6t %N", "--noheader"],
        timeout=10,
    )
    if not out["ok"] or not out["stdout"].strip():
        return {"available": True, "error": out["stderr"].strip() or "no output"}
    rows = []
    for line in out["stdout"].strip().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            rows.append({
                "partition": parts[0].rstrip("*"),
                "avail":     parts[1],
                "timelimit": parts[2],
                "nodes":     parts[3],
                "state":     parts[4],
                "nodelist":  parts[5],
            })
        elif parts:
            rows.append({"raw": line.strip()})
    return {"available": True, "partitions": rows}


@lru_cache(maxsize=1)
def _collect_modules() -> dict:
    """What is loaded now, plus a pointer to the searchable catalogue.

    This used to return `module -t avail | head -n 120` truncated to 80 names under a
    key called "sample". An alphabetical slice of a module tree is worse than no list
    at all: a model that reads "abaqus … cmake" with no cuda in sight concludes CUDA
    is unavailable. The full tree is searched with platform_search; what belongs here
    is only the volatile part, which is the set of modules actually loaded.

    Never triggers a build — platform_probe must not inherit the catalogue's cost.
    """
    flavour = _module_flavour()
    if flavour == "none":
        return {"available": False, "module_system": "none",
                "note": "No Lmod or Tcl Environment Modules on this host."}
    loaded = _run_shell(module_probe_script("module -t list 2>&1"), timeout=15)
    catalogue = _load_catalogue()
    summary = {"indexed": False}
    if catalogue:
        summary = {
            "indexed": True, "count": catalogue.get("count", 0),
            "tier": catalogue.get("tier", ""), "built_at": catalogue.get("built_at", ""),
            "partial": catalogue.get("partial", False),
        }
    return {
        "available": True,
        "module_system": flavour,
        "loaded": [e["load"] for e in _parse_avail_terse(loaded["stdout"] or "")],
        "catalogue": summary,
        "digest": (catalogue or {}).get("digest", {}),
        "note": "The module catalogue is not listed here — it is searchable with "
                "platform_search(query), by name ('cuda') or by capability "
                "('parallel hdf5'). Modules shown above are the ones loaded right now.",
    }


@lru_cache(maxsize=1)
def _collect_toolchains() -> dict:
    tools = {}
    # Vendor-plural on purpose: a site may ship GNU, LLVM, Intel oneAPI, the NVIDIA
    # HPC SDK, AMD ROCm or Cray wrappers, and an absent one simply does not appear.
    for name in [
        "gcc", "g++", "gfortran",
        "clang", "clang++", "flang",
        "icx", "icpx", "ifx",
        "nvc", "nvc++", "nvfortran", "nvcc",
        "hipcc",
        "cc", "CC", "ftn",
        "mpicc", "mpicxx", "mpifort",
        "make", "cmake", "ninja",
        "python3", "pytest", "ruff", "mypy",
    ]:
        if _cmd_exists(name):
            out = _run([name, "--version"])
            first = (out["stdout"].splitlines() or out["stderr"].splitlines() or [""])[0]
            tools[name] = first.strip()
    return tools

# ── module catalogue ──────────────────────────────────────────────────────────
# The site's module tree is hundreds to thousands of entries: it does not fit in a
# context window, so it is indexed once and searched, never dumped. Everything below
# serves that index; every *live* fact above stays live.

_CATALOGUE_SCHEMA = 1

# Named by hostname: STATE_DIR is per-workspace, but a module tree belongs to a
# machine. On a site whose state dir sits on a shared filesystem, a single file would
# be rebuilt on every hop between login nodes.
_MODULES_DIR = os.environ.get(
    "MIMIR_MODULE_CATALOGUE_DIR", os.path.join(state_dir(), "platform", "modules")
)
_CATALOGUE_FILE = os.path.join(_MODULES_DIR, f"catalogue-{socket.gethostname()}.json")
_EMBEDDINGS_FILE = os.path.join(_MODULES_DIR, f"embeddings-{socket.gethostname()}.json")

# Lmod invocations for tier B cost ~30 ms each; past this many the catalogue keeps its
# name-only entries and says so rather than spending minutes on a 5000-module site.
_WHATIS_MAX = 800
# Bound on the stat walk behind the freshness signal.
_TREE_SCAN_MAX = 20000
# How many lexically-plausible modules get embedded for the semantic rerank. Memory
# embeds its whole corpus; a module catalogue must not, or the first search pays for
# thousands of texts.
_SEMANTIC_POOL = 200
# The background enrichment's wall clock. Generous on purpose: nobody is waiting on it.
_ENRICH_BUDGET_SECS = int(os.environ.get("MIMIR_MODULE_INDEX_BUDGET", "600"))
# A manual refresh arriving inside this window is a no-op. Not an invalidation TTL —
# the signal alone decides freshness — just a debounce so a model that takes to
# passing refresh=True cannot stack enrichment threads.
_REFRESH_DEBOUNCE_SECS = 60

# Re-entrant so a test that runs the "background" work inline does not deadlock.
_BUILD_LOCK = threading.RLock()
_CATALOGUE_CACHE: dict | None = None
_ENRICHING = False
_LAST_BUILD_MONOTONIC: float | None = None


def _schedule_background(fn) -> None:
    """Run *fn* off the request path. Replaced in tests to run it inline."""
    threading.Thread(target=fn, daemon=True).start()


@lru_cache(maxsize=1)
def _module_flavour() -> str:
    """"lmod", "tmod", or "none" — no module system here, which is not an error."""
    script = module_probe_script(
        'if ! type module >/dev/null 2>&1; then echo none; '
        'elif [ -n "${LMOD_CMD:-}" ] || module --version 2>&1 | grep -qi lua; then echo lmod; '
        'else echo tmod; fi'
    )
    out = _run_shell(script, timeout=15)
    answer = (out["stdout"] or "").strip().splitlines()
    value = answer[-1].strip() if answer else ""
    return value if value in ("lmod", "tmod", "none") else "none"


@lru_cache(maxsize=1)
def _effective_modulepath() -> str:
    """MODULEPATH as the site's init scripts leave it, not as it reaches this process.

    Memoized: this server never runs `module use` on itself, so the value cannot
    change while the process lives, and the check would otherwise spend a subprocess
    on every single search.
    """
    out = _run_shell(module_probe_script('echo "${MODULEPATH:-}"'), timeout=15)
    lines = [ln.strip() for ln in (out["stdout"] or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _lmod_cache_files() -> list[str]:
    """Lmod's own spider cache. Where it exists it is the most authoritative
    "the module tree changed" signal there is: the site regenerates it exactly then."""
    found = []
    roots = []
    mp_root = os.environ.get("MODULEPATH_ROOT", "")
    if mp_root:
        roots.append(os.path.join(os.path.dirname(mp_root.rstrip("/")), "cache"))
    roots.append(os.path.join(os.path.expanduser("~"), ".lmod.d", ".cache"))
    for root in roots:
        try:
            for name in sorted(os.listdir(root)):
                if name.startswith("spiderT"):
                    found.append(os.path.join(root, name))
        except OSError:
            continue
    return found


def _tree_fingerprint(modulepath: str) -> dict:
    """A deterministic hash of the module tree's identity — never a timestamp.

    Depth two (family, then version) catches every real mutation: installing
    cuda/12.4 changes .../cuda's mtime, adding a family changes the root's. The
    residual blind spot is a modulefile edited in place without touching any
    directory mtime; the Lmod cache below closes it wherever Lmod is in use.
    """
    parts: list[str] = []
    seen = 0
    truncated = False
    for root in [p for p in modulepath.split(":") if p]:
        try:
            st = os.stat(root)
        except OSError:
            continue
        parts.append(f"{root}|{st.st_mtime_ns}|{st.st_size}")
        try:
            families = sorted(os.scandir(root), key=lambda e: e.name)
        except OSError:
            continue
        for fam in families:
            if seen >= _TREE_SCAN_MAX:
                truncated = True
                break
            seen += 1
            try:
                parts.append(f"{fam.path}|{fam.stat().st_mtime_ns}")
                if not fam.is_dir():
                    continue
                for ver in sorted(os.scandir(fam.path), key=lambda e: e.name):
                    if seen >= _TREE_SCAN_MAX:
                        truncated = True
                        break
                    seen += 1
                    parts.append(f"{ver.path}|{ver.stat().st_mtime_ns}")
            except OSError:
                continue
        if truncated:
            break
    for cache_file in _lmod_cache_files():
        try:
            parts.append(f"{cache_file}|{os.stat(cache_file).st_mtime_ns}")
        except OSError:
            continue
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return {"tree": digest, "truncated": truncated}


def _catalogue_signal(flavour: str) -> dict:
    modulepath = _effective_modulepath()
    fingerprint = _tree_fingerprint(modulepath)
    return {
        "schema": _CATALOGUE_SCHEMA,
        "hostname": socket.gethostname(),
        "flavour": flavour,
        "modulepath": modulepath,
        **fingerprint,
    }


# ── tier parsers ──────────────────────────────────────────────────────────────

# Lmod decorates a name with (D)efault, (L)oaded, and friends.
_AVAIL_MARKER_RE = re.compile(r"\(([A-Za-z],?)+\)$")
# Chatter Lmod prints among the names when stdout and stderr are merged.
_AVAIL_NOISE = ("Where:", "Use \"module", "Use 'module", "If the avail list",
                "To learn more", "The following")


def _split_name(load: str) -> tuple[str, str]:
    """'cuda/12.2' -> ('cuda', '12.2'); a bare 'cmake' -> ('cmake', '')."""
    if "/" in load:
        head, _, tail = load.rpartition("/")
        return head, tail
    return load, ""


def _parse_avail_terse(stdout: str) -> list[dict]:
    """Parse `module -t avail` with stderr merged in.

    The merge is deliberate: Lmod writes the MODULEPATH section headers to stderr and
    the names to stdout, so merging is the only way to attribute a name to its tree.
    Headers are therefore *consumed* (they set the current root), never just dropped.
    """
    entries: list[dict] = []
    seen: set[str] = set()
    root = ""
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("-"):
            continue
        if any(line.startswith(prefix) for prefix in _AVAIL_NOISE):
            continue
        if line.endswith(":"):
            candidate = line[:-1].strip()
            if candidate.startswith("/") or os.path.isdir(candidate):
                root = candidate
            continue
        default = False
        marker = _AVAIL_MARKER_RE.search(line)
        if marker:
            flags = marker.group(0).strip("()").split(",")
            default = any(f.strip().upper() in ("D", "DEFAULT") for f in flags)
            line = line[:marker.start()].strip()
        if not line or " " in line:
            continue
        if line in seen:
            if default:
                for e in entries:
                    if e["load"] == line:
                        e["default"] = True
            continue
        seen.add(line)
        name, version = _split_name(line)
        entries.append({
            "kind": "module",
            "key": line,
            "load": line,
            "name": name,
            "version": version,
            "default": default,
            "description": "",
            "category": "",
            "keywords": [],
            "source": os.path.join(root, line) if root else "",
            "tier": "name",
        })
    return entries


def _whatis_description(lines: list[str], load: str) -> str:
    """Squeeze a one-line description out of whatis output, whatever shape it took.

    Lmod and Tcl Modules disagree on the layout, and sites add their own; rather than
    match a format, take the text after the first colon on each line, drop the parts
    that only repeat the module's own name, and join what is left.
    """
    pieces = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if ":" in text:
            head, _, tail = text.partition(":")
            if head.strip() in (load, load.split("/")[0], "Description", "description"):
                text = tail.strip()
            elif head.strip().startswith(load.split("/")[0]):
                text = tail.strip()
        if text.lower().startswith("description:"):
            text = text.split(":", 1)[1].strip()
        if not text or text == load:
            continue
        pieces.append(text)
    return " ".join(pieces)[:400]


def _parse_whatis_bulk(stdout: str, names: list[str]) -> dict:
    """Parse the '@@<name>'-delimited whatis loop into {load: description}."""
    out: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for raw in stdout.splitlines():
        if raw.startswith("@@"):
            if current is not None:
                out[current] = _whatis_description(buffer, current)
            current = raw[2:].strip()
            buffer = []
            continue
        buffer.append(raw)
    if current is not None:
        out[current] = _whatis_description(buffer, current)
    return {k: v for k, v in out.items() if v}


def _parse_whatis_flat(stdout: str, known: set[str]) -> dict:
    """Parse Tcl Modules' argument-less `module whatis`, one line per module."""
    out: dict[str, str] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        head, _, tail = line.partition(":")
        load = head.strip()
        if load in known and tail.strip():
            out[load] = tail.strip()[:400]
    return out


def _spider_versions(package: dict) -> list[dict]:
    versions = package.get("versions")
    return versions if isinstance(versions, list) else []


def _parse_spider_json(stdout: str) -> dict:
    """Parse either spider output shape into {load: {description, category, ...}}.

    `-o jsonSoftwarePage` yields a list of packages; `-o spider-json` a mapping of
    name to modulefile records. The shapes differ across Lmod majors and sites run
    everything from 6 to 8.x, so anything unrecognized returns {} and the caller
    falls to the next tier — never an exception.
    """
    try:
        data = json.loads(stdout)
    except (TypeError, ValueError):
        return {}

    out: dict[str, dict] = {}

    def _record(load, description, path, category, keywords, default):
        if not load:
            return
        out[load] = {
            "description": (description or "").strip()[:400],
            "source": path or "",
            "category": category or "",
            "keywords": keywords or [],
            "default": bool(default),
        }

    if isinstance(data, list):
        for package in data:
            if not isinstance(package, dict):
                continue
            base_desc = package.get("description") or ""
            categories = package.get("categories") or package.get("category") or ""
            if isinstance(categories, list):
                categories = ", ".join(str(c) for c in categories)
            keywords = package.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []
            for version in _spider_versions(package):
                if not isinstance(version, dict):
                    continue
                _record(
                    version.get("full") or version.get("fullName"),
                    version.get("description") or base_desc,
                    version.get("path") or version.get("pathToDB"),
                    categories,
                    [str(k) for k in keywords],
                    version.get("markedDefault") or version.get("default"),
                )
        return out

    if isinstance(data, dict):
        for _name, by_path in data.items():
            if not isinstance(by_path, dict):
                continue
            for path, record in by_path.items():
                if not isinstance(record, dict):
                    continue
                keywords = record.get("keywords") or record.get("propT") or []
                if not isinstance(keywords, list):
                    keywords = []
                _record(
                    record.get("fullName") or record.get("full"),
                    record.get("Description") or record.get("description"),
                    record.get("path") or path,
                    record.get("category") or "",
                    [str(k) for k in keywords],
                    record.get("markedDefault") or record.get("default"),
                )
        return out

    return {}


# ── tier collectors ───────────────────────────────────────────────────────────

def _collect_tier_c() -> list[dict]:
    """Names only. The floor: fast, and enough for a search by module name."""
    out = _run_shell(module_probe_script("module -t avail 2>&1"), timeout=30)
    return _parse_avail_terse(out["stdout"] or "")


def _collect_tier_a(timeout: int) -> dict:
    """Lmod's spider. Never attempted on Tcl Environment Modules, which has no such
    command; the tier exists only where it exists."""
    modulepath = _effective_modulepath()
    if not modulepath:
        return {}
    for fmt in ("jsonSoftwarePage", "spider-json"):
        script = module_probe_script(
            'SPIDER="${LMOD_DIR:-${LMOD_PKG:-}/libexec}/spider"; '
            f'[ -x "$SPIDER" ] || SPIDER=spider; "$SPIDER" -o {fmt} "$MODULEPATH" 2>/dev/null'
        )
        out = _run_shell(script, timeout=timeout)
        parsed = _parse_spider_json(out["stdout"] or "")
        if parsed:
            return parsed
    return {}


def _collect_tier_b(flavour: str, names: list[str], timeout: int) -> dict:
    """Descriptions via whatis. One shell process either way, never one per module."""
    if not names:
        return {}
    if flavour == "tmod":
        out = _run_shell(module_probe_script("module whatis 2>&1"), timeout=timeout)
        return _parse_whatis_flat(out["stdout"] or "", set(names))
    payload = "\n".join(names)
    script = module_probe_script(
        "while IFS= read -r m; do printf '@@%s\\n' \"$m\"; "
        "module whatis \"$m\" 2>&1; done <<'MIMIR_EOF'\n" + payload + "\nMIMIR_EOF"
    )
    out = _run_shell(script, timeout=timeout)
    return _parse_whatis_bulk(out["stdout"] or "", names)


# ── the digest ────────────────────────────────────────────────────────────────

def _collect_partitions() -> list[dict]:
    if not _cmd_exists("sinfo"):
        return []
    out = _run(["sinfo", "-h", "-o", "%P|%a|%l|%D|%c|%m"], timeout=10)
    if not out["ok"]:
        return []
    rows = []
    for line in (out["stdout"] or "").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 6:
            rows.append({
                "partition": cells[0].rstrip("*"),
                "availability": cells[1],
                "time_limit": cells[2],
                "nodes": cells[3],
                "cpus_per_node": cells[4],
                "mem_mb_per_node": cells[5],
            })
    return rows


def _scontrol_nodes() -> list[dict]:
    if not _cmd_exists("scontrol"):
        return []
    out = _run(["scontrol", "show", "node", "-o"], timeout=15)
    if not out["ok"]:
        return []
    return parse_scontrol_nodes(out["stdout"] or "")


def _build_digest() -> dict:
    """The stable, high-impact facts that *fit* in a context window.

    Not searched — injected. This is what lets the agent know, without spending a
    tool call, that the compute nodes are a different architecture from the login
    node it is running on, which is the difference between a binary that runs and one
    that dies on an illegal instruction. Occupancy is deliberately absent: it is
    volatile, slurm_nodes owns it, and including it would make the signal below
    change every few seconds.
    """
    nodes = _scontrol_nodes()
    cpu = _collect_cpu()
    return {
        "built_at": now_iso(),
        "host": {
            "hostname": socket.gethostname(),
            "arch": cpu.get("arch", ""),
            "model": cpu.get("model", ""),
            "logical_cpus": cpu.get("logical_cpus"),
            "simd": cpu.get("simd", {}),
            "memory": _collect_memory(),
        },
        "node_types": stable_types(aggregate_node_types(nodes)) if nodes else [],
        "partitions": _collect_partitions(),
        "toolchains": _collect_toolchains(),
        "signature": stable_signature(nodes) if nodes else "",
    }


# ── catalogue persistence ─────────────────────────────────────────────────────

def _load_catalogue() -> dict | None:
    """Read the catalogue from disk. Never builds — platform_probe depends on that."""
    try:
        with open(_CATALOGUE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == _CATALOGUE_SCHEMA else None


def _save_catalogue(catalogue: dict) -> None:
    """Atomic: a killed build leaves the previous file, never half a JSON document.
    A read-only state dir degrades to rebuilding each session, not to an error."""
    tmp = _CATALOGUE_FILE + ".tmp"
    try:
        os.makedirs(_MODULES_DIR, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, ensure_ascii=False)
        os.replace(tmp, _CATALOGUE_FILE)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _empty_catalogue(flavour: str) -> dict:
    return {
        "schema": _CATALOGUE_SCHEMA, "built_at": now_iso(),
        "hostname": socket.gethostname(), "flavour": flavour,
        "tier": "none", "partial": False, "enrich_done": True,
        "signal": {}, "count": 0, "modules": [], "digest": {},
    }


def _enrich(base: dict) -> None:
    """Fill in descriptions behind the user's back, then swap the catalogue.

    Runs off the request path so nobody waits for spider. Its result is discarded if
    the tree moved while it worked — the next call will rebuild against what is
    actually there now.
    """
    global _ENRICHING, _CATALOGUE_CACHE, _LAST_BUILD_MONOTONIC
    try:
        started = time.monotonic()
        flavour = base.get("flavour", "")
        entries = [dict(e) for e in base.get("modules", [])]
        by_load = {e["load"]: e for e in entries}
        tier = "name"

        if flavour == "lmod":
            remaining = max(5, _ENRICH_BUDGET_SECS - int(time.monotonic() - started))
            for load, info in _collect_tier_a(min(120, remaining)).items():
                entry = by_load.get(load)
                if entry is None:
                    continue
                entry.update({
                    "description": info["description"],
                    "category": info["category"],
                    "keywords": info["keywords"],
                    "source": info["source"] or entry["source"],
                    "tier": "spider",
                })
                if info["default"]:
                    entry["default"] = True
                tier = "spider"

        pending = [e["load"] for e in entries if e["tier"] == "name"][:_WHATIS_MAX]
        if pending:
            remaining = max(5, _ENRICH_BUDGET_SECS - int(time.monotonic() - started))
            for load, description in _collect_tier_b(flavour, pending, remaining).items():
                entry = by_load.get(load)
                if entry is not None and description:
                    entry["description"] = description
                    entry["tier"] = "whatis"
                    if tier == "name":
                        tier = "whatis"

        digest = _build_digest()

        if _catalogue_signal(flavour) != base.get("signal"):
            return  # the tree moved under us; the next call rebuilds

        enriched = {
            **base,
            "built_at": now_iso(),
            "tier": tier,
            "partial": any(e["tier"] == "name" for e in entries),
            "enrich_done": True,
            "modules": entries,
            "count": len(entries),
            "digest": digest,
        }
        with _BUILD_LOCK:
            _CATALOGUE_CACHE = enriched
            _LAST_BUILD_MONOTONIC = time.monotonic()
            _save_catalogue(enriched)
    except Exception:
        # A failed enrichment must never take the tier-C catalogue down with it.
        pass
    finally:
        _ENRICHING = False


def _build_tier_c(flavour: str, signal: dict) -> dict:
    entries = _collect_tier_c()
    return {
        "schema": _CATALOGUE_SCHEMA,
        "built_at": now_iso(),
        "hostname": socket.gethostname(),
        "flavour": flavour,
        "tier": "name",
        "partial": True,
        "enrich_done": False,
        "signal": signal,
        "count": len(entries),
        "modules": entries,
        "digest": {},
    }


def _ensure_catalogue(refresh: bool = False) -> dict:
    """The catalogue for this host, building only what the caller must wait for.

    Synchronous work is tier C alone (under two seconds); descriptions arrive in the
    background. A hit on the in-process memo or on disk costs neither.
    """
    global _CATALOGUE_CACHE, _ENRICHING, _LAST_BUILD_MONOTONIC

    flavour = _module_flavour()
    if flavour == "none":
        return _empty_catalogue(flavour)

    signal = _catalogue_signal(flavour)
    with _BUILD_LOCK:
        debounced = (
            _LAST_BUILD_MONOTONIC is not None
            and (time.monotonic() - _LAST_BUILD_MONOTONIC) < _REFRESH_DEBOUNCE_SECS
        )
        forced = refresh and not debounced and not _ENRICHING

        catalogue = _CATALOGUE_CACHE
        if forced or catalogue is None or catalogue.get("signal") != signal:
            catalogue = None if forced else _load_catalogue()
            if catalogue is not None and catalogue.get("signal") == signal:
                _CATALOGUE_CACHE = catalogue
            else:
                catalogue = _build_tier_c(flavour, signal)
                _CATALOGUE_CACHE = catalogue
                _LAST_BUILD_MONOTONIC = time.monotonic()
                _save_catalogue(catalogue)

        if not catalogue.get("enrich_done") and not _ENRICHING:
            _ENRICHING = True
            base = catalogue
            _schedule_background(lambda: _enrich(base))

        return catalogue


# ── ranking ───────────────────────────────────────────────────────────────────

def _natural_version(version: str) -> tuple:
    """Sort 12.10 above 12.9, and never raise on a version that is not numbers."""
    chunks = []
    for part in re.findall(r"\d+|\D+", version or ""):
        if part.isdigit():
            chunks.append((0, int(part), ""))
        else:
            chunks.append((1, 0, part))
    return tuple(chunks)


def _search_text(entry: dict) -> str:
    return " ".join([
        entry.get("name", ""), entry.get("version", ""),
        entry.get("description", ""), entry.get("category", ""),
        " ".join(entry.get("keywords", [])),
    ]).strip()


def _name_matches(modules: list[dict], query: str) -> list[dict]:
    """Exact, then prefix, then substring; defaults first, newest version first.

    This layer is what makes a description-free, embedding-free catalogue genuinely
    useful rather than merely non-broken, so it runs first and always.
    """
    q = query.strip().lower()
    if not q:
        return []
    ranked = []
    for entry in modules:
        name = entry.get("name", "").lower()
        load = entry.get("load", "").lower()
        if name == q or load == q:
            rank = 0
        elif name.startswith(q) or load.startswith(q):
            rank = 1
        elif q in name or q in load:
            rank = 2
        else:
            continue
        ranked.append((rank, entry))
    ranked.sort(key=lambda pair: _natural_version(pair[1].get("version", "")), reverse=True)
    ranked.sort(key=lambda pair: (pair[0], 0 if pair[1].get("default") else 1))
    return [entry for _rank, entry in ranked]


def _rank_modules(modules: list[dict], query: str, limit: int) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()

    for entry in _name_matches(modules, query):
        if entry["key"] in seen:
            continue
        seen.add(entry["key"])
        results.append({**entry, "score": None, "match": "name"})
        if len(results) >= limit:
            return results

    described = [e for e in modules if e.get("description") or e.get("keywords")]
    if not described:
        return results

    texts = [_search_text(e) for e in described]
    pool = [described[idx] for idx, overlap in _embed.lexical_rank(query, texts)
            if overlap > 0][:_SEMANTIC_POOL]
    if not pool:
        return results

    semantic = vector_cache.semantic_rank(
        query, pool,
        key_of=lambda e: e["key"],
        text_of=_search_text,
        path=_EMBEDDINGS_FILE,
        limit=limit,
    )
    if semantic is not None:
        ordered = [(entry, score, "semantic") for entry, score in semantic]
    else:
        ordered = [(entry, None, "lexical") for entry in pool]

    for entry, score, kind in ordered:
        if entry["key"] in seen:
            continue
        seen.add(entry["key"])
        results.append({**entry, "score": score, "match": kind})
        if len(results) >= limit:
            break
    return results


def _collect_conda_envs() -> dict:
    """Collect available Conda environments (read-only).

    Works on typical HPC setups where conda is available as a module or binary.
    Does not activate or modify any environment.
    """
    if not _cmd_exists("conda"):
        return {"available": False}

    # Use `conda env list --json` for robust parsing
    out = _run(["conda", "env", "list", "--json"], timeout=10)
    if not out["ok"]:
        return {
            "available": True,
            "error": out["stderr"].strip() or "failed to query conda env list",
        }

    try:
        data = json.loads(out["stdout"])
    except Exception as exc:
        return {
            "available": True,
            "error": f"invalid json output: {exc}",
        }

    envs = []
    for path in data.get("envs", []):
        name = os.path.basename(path)
        envs.append({
            "name": name,
            "path": path,
            "python": os.path.join(path, "bin", "python"),
        })

    return {
        "available": True,
        "count": len(envs),
        "envs": envs,
    }


def _is_python_exec(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK) and os.path.basename(path).startswith("python")

def _collect_virtualenvs(workspace_root: str | None = None) -> dict:
    """Discover Python virtualenvs without activating them."""
    candidates = []

    roots = []
    if workspace_root:
        roots.extend([
            os.path.join(workspace_root, ".venv"),
            os.path.join(workspace_root, "venv"),
            os.path.join(workspace_root, "env"),
        ])

    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".virtualenvs"))

    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        if os.path.basename(root) in {".venv", "venv", "env"}:
            py = os.path.join(root, "bin", "python")
            if _is_python_exec(py):
                candidates.append({
                    "kind": "virtualenv",
                    "path": root,
                    "python": py,
                    "source": "project" if workspace_root and root.startswith(workspace_root) else "user",
                })
            continue

        # ~/.virtualenvs/*
        for name in os.listdir(root):
            venv_dir = os.path.join(root, name)
            py = os.path.join(venv_dir, "bin", "python")
            if py in seen:
                continue
            if _is_python_exec(py):
                seen.add(py)
                candidates.append({
                    "kind": "virtualenv",
                    "path": venv_dir,
                    "python": py,
                    "source": "user",
                })

    return {
        "available": bool(candidates),
        "count": len(candidates),
        "envs": candidates,
    }

def _build_profile() -> dict:
    workspace_root = os.getcwd()
    return {
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": _collect_cpu(),
        "memory": _collect_memory(),
        "gpu": _collect_gpu(),
        "slurm": _collect_slurm(),
        "modules": _collect_modules(),
        "toolchains": _collect_toolchains(),
        "conda_envs": _collect_conda_envs(),
        "virtualenvs": _collect_virtualenvs(workspace_root),
    }



@mcp.tool(**tool_caps(caps=[ENV_DISCOVERY]))
def platform_probe() -> dict:
    """Collect and return a fresh platform profile.

    Every fact is probed on demand, so none of it can be stale. It reads the module
    catalogue's summary if one already exists but never builds it — use
    platform_search to search the modules themselves.
    """
    started = time.perf_counter()
    profile = _build_profile()
    elapsed = time.perf_counter() - started
    return ok({"profile": profile, "elapsed_s": round(elapsed, 3)})


@mcp.tool(**tool_caps(caps=[ENV_DISCOVERY]))
def platform_get_profile() -> dict:
    """Return a fresh platform profile for the current host, with live sinfo data.

    Built on demand for the current host, so it is always correct rather than
    remembered. Like platform_probe, it reports the module catalogue's summary
    without building it.
    """
    return ok({"profile": _build_profile(), "sinfo": _collect_sinfo()})


@mcp.tool(**tool_caps(
    caps=[ENV_DISCOVERY, CACHEABLE],
    label="Searching modules: {query}",
))
def platform_search(query: str, limit: int = 10, refresh: bool = False) -> dict:
    """Search this host's environment-module catalogue by name or by what a module does.

    A site's module tree is hundreds to thousands of entries — far more than fits in a
    context window — so it is indexed once and searched here instead of listed. Ask by
    name ("cuda", "openmpi") or by capability ("parallel hdf5", "fortran compiler with
    openmp"). Every hit carries `load`: the exact string to pass to `module load`.

    The index is built on first use. Only the name-level pass is built while you wait
    (a second or two); descriptions are filled in behind you, so a result carrying
    `catalogue.enriching: true` means an empty description is *not yet known* rather
    than absent. It is then reused until the module tree itself changes.

    Hosts with no module system return an empty result, not an error.

    Args:
        query: A module name, or a description of what you need it to do.
        limit: Maximum hits to return (1-50).
        refresh: Force a rebuild of the index. Only when the user explicitly asks —
            the index invalidates itself when the module tree changes.
    """
    if not isinstance(query, str) or not query.strip():
        return ok({"query": query, "modules": [], "count": 0,
                   "note": "Give a module name or a description of what you need."})
    limit = 10 if not isinstance(limit, int) or limit <= 0 else min(limit, 50)

    catalogue = _ensure_catalogue(refresh=bool(refresh))
    flavour = catalogue.get("flavour", "none")
    if flavour == "none":
        return ok({
            "query": query, "module_system": "none", "modules": [], "count": 0,
            "note": "No Lmod or Tcl Environment Modules on this host, so there is no "
                    "module catalogue to search. Software here is whatever is on PATH "
                    "or in a Python environment.",
        })

    modules = _rank_modules(catalogue.get("modules", []), query, limit)
    payload = {
        "query": query,
        "module_system": flavour,
        # Named "modules", not "results": a payload carrying "results" is read as the
        # end of workspace discovery by the client's workflow observer, and finding a
        # module says nothing about having explored the repository.
        "modules": modules,
        "count": len(modules),
        "total_indexed": catalogue.get("count", 0),
        "catalogue": {
            "tier": catalogue.get("tier", ""),
            "partial": catalogue.get("partial", False),
            "built_at": catalogue.get("built_at", ""),
            "enriching": not catalogue.get("enrich_done", False),
        },
    }
    if modules:
        payload["hint"] = "Load one with: module load <load>"
    return ok(payload)


@mcp.tool(**tool_caps(caps=[ENV_DISCOVERY, CACHEABLE]))
def platform_catalogue_status() -> dict:
    """Report the module catalogue's state without building or refreshing it.

    Says which host it describes, which module system, how many entries, how detailed
    they are, and whether enrichment is still running.
    """
    flavour = _module_flavour()
    catalogue = _load_catalogue()
    if catalogue is None:
        return ok({
            "module_system": flavour, "indexed": False,
            "path": _CATALOGUE_FILE,
            "note": ("No module system on this host." if flavour == "none"
                     else "Not indexed yet; the first platform_search builds it."),
        })
    fresh = flavour != "none" and catalogue.get("signal") == _catalogue_signal(flavour)
    return ok({
        "module_system": flavour,
        "indexed": True,
        "path": _CATALOGUE_FILE,
        "hostname": catalogue.get("hostname", ""),
        "count": catalogue.get("count", 0),
        "tier": catalogue.get("tier", ""),
        "partial": catalogue.get("partial", False),
        "enriching": _ENRICHING,
        "built_at": catalogue.get("built_at", ""),
        "signal_fresh": fresh,
        "digest": catalogue.get("digest", {}),
    })


if __name__ == "__main__":
    mcp.run()
