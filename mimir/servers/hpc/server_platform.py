"""
MCP Platform Server
===================
Reports what this host actually is — CPU/SIMD, memory, GPU, Slurm, modules,
toolchains, Python environments. Facts only: the architecture-aware *advice* that
used to live here was a frozen lookup table the model already knows better, paid for
with a full hardware probe per call. Stateless: nothing is persisted to disk; the
collectors whose answer cannot change mid-process are memoized.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import json
from datetime import datetime, timezone
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from module_env import module_shell_script
from capabilities import tool_caps, ENV_DISCOVERY
from responses import err, ok

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
    query = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    out = _run_shell(query)
    if not out["ok"]:
        return {"available": False, "vendors": present, "error": out["stderr"].strip()}
    gpus = []
    for line in out["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
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
    script = module_shell_script("module -t avail 2>&1 | head -n 120")
    out = _run_shell(script)
    if not out["ok"]:
        return {"available": False, "error": out["stderr"].strip()}
    lines = [ln.strip() for ln in out["stdout"].splitlines() if ln.strip()]
    return {"available": True, "sample": lines[:80], "count_sample": len(lines[:80])}


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

    Stateless: the profile is built on demand and returned to the caller; nothing is
    written to disk.
    """
    started = time.perf_counter()
    profile = _build_profile()
    elapsed = time.perf_counter() - started
    return ok({"profile": profile, "elapsed_s": round(elapsed, 3)})


@mcp.tool(**tool_caps(caps=[ENV_DISCOVERY]))
def platform_get_profile() -> dict:
    """Return a fresh platform profile for the current host, with live sinfo data.

    Stateless: the profile is built on demand for the current host (so it is always
    correct, never stale) and returned to the caller; nothing is persisted.
    """
    return ok({"profile": _build_profile(), "sinfo": _collect_sinfo()})


if __name__ == "__main__":
    mcp.run()
