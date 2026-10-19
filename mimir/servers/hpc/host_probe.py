"""What this host is, probed on demand: CPU, ``-march=native``, memory, GPU, Slurm's tools,
toolchains, Python environments, and the digest of the stable facts (the cluster half
from ``slurm_digest``). Probes run in the C locale, never on the server's MCP pipe.
"""

import json
import os
import platform
import shutil
import socket
import subprocess
from functools import lru_cache
from typing import Any

from ...shared.clock import utc_stamp
from ...shared.cpu_facts import (
    GPU_FIELDS,
    GPU_FIELDS_BASE,
    GPU_PROBES,
    GPU_QUERY,
    GPU_QUERY_BASE,
    MARCH_QUERY,
    cpu_facts,
    parse_march,
    parse_nvidia_csv,
)
from ...shared.slurm_site import hosts as slurm_hosts, name_of as _host_name
from .slurm_digest import cached_cluster_digest


def cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


# Probes run in the C locale: lscpu and free translate their field names ("Nom de
# modèle", "Drapeaux"), and a parser keyed on the English names then reads a
# French-locale host as having no CPU model and no SIMD at all.
_PROBE_ENV = {**os.environ, "LC_ALL": "C"}


def run(cmd: list[str], timeout: int = 8) -> dict:
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,   # never the server's MCP pipe
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=_PROBE_ENV,
        )
        return {
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(e)}


def run_shell(script: str, timeout: int = 10) -> dict:
    return run(["bash", "-lc", script], timeout=timeout)


@lru_cache(maxsize=1)
def collect_cpu() -> dict:
    data: dict[str, Any] = {
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if cmd_exists("lscpu"):
        out = run(["lscpu"])
        if out["ok"]:
            data.update(cpu_facts(data["arch"], out["stdout"]))
    return data


@lru_cache(maxsize=1)
def collect_march() -> dict:
    """What ``-march=native`` means on this host, as a name usable elsewhere."""
    if not cmd_exists("gcc"):
        return {}
    out = run_shell(MARCH_QUERY)
    return parse_march(out["stdout"]) if out["ok"] else {}


def collect_memory() -> dict:
    data = {}
    if cmd_exists("free"):
        out = run(["free", "-b"])
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


# Only the NVIDIA output is parsed into devices; the others are detected and reported
# as such, because claiming "no GPU" on a machine whose accelerator this probe cannot
# read is worse than saying so. Cluster-wide GPU truth comes from Slurm GRES
# (slurm_nodes), which is vendor-neutral. Query strings and fields: cpu_facts.


@lru_cache(maxsize=1)
def collect_gpu() -> dict:
    present = [vendor for vendor, cmd in GPU_PROBES if cmd_exists(cmd)]
    if not present:
        return {"available": False, "probed": [cmd for _, cmd in GPU_PROBES]}
    if "nvidia" not in present:
        return {
            "available": True, "vendors": present, "devices": [],
            "note": "Accelerator detected but not enumerated: only the NVIDIA probe is "
                    "parsed here. Ask Slurm (slurm_nodes) for GPU type and count.",
        }
    out = run_shell(GPU_QUERY)
    fields: tuple[str, ...] = GPU_FIELDS
    if not out["ok"]:
        # An nvidia-smi too old to know a field rejects the whole query rather than
        # the field, which would cost us the enumeration entirely. Retry without the
        # newer field before reporting the host as unreadable.
        out = run_shell(GPU_QUERY_BASE)
        fields = GPU_FIELDS_BASE
    if not out["ok"]:
        return {"available": False, "vendors": present, "error": out["stderr"].strip()}
    gpus = parse_nvidia_csv(out["stdout"], fields)
    return {"available": bool(gpus), "vendors": present, "count": len(gpus), "devices": gpus}


@lru_cache(maxsize=1)
def collect_slurm() -> dict:
    """Whether Slurm's tools are on this host, plus the schedulers this workspace is known to reach."""
    if not cmd_exists("sinfo"):
        elsewhere = [_host_name(host) for host in slurm_hosts() if host]
        if elsewhere:
            return {
                "available": False,
                "schedulers": elsewhere,
                "note": ("Not on this host. Schedulers have answered through "
                         + ", ".join(elsewhere) + ", and the cluster tools submit to "
                         "them by name; this profile describes this machine only."),
            }
        return {"available": False}
    out = run(["sinfo", "--version"])
    if not out["ok"]:
        return {"available": False, "error": out["stderr"].strip()}
    version = out["stdout"].strip() or out["stderr"].strip()
    return {"available": True, "version": version}


def collect_sinfo() -> dict:
    if not cmd_exists("sinfo"):
        return {"available": False}
    out = run(
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
def collect_toolchains() -> dict:
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
        if cmd_exists(name):
            out = run([name, "--version"])
            first = (out["stdout"].splitlines() or out["stderr"].splitlines() or [""])[0]
            tools[name] = first.strip()
    return tools


# ── the digest ────────────────────────────────────────────────────────────────

def build_digest() -> dict:
    """The stable facts that fit in a context: this host, and the cluster from ``slurm_digest``.

    No occupancy, which is volatile and ``slurm_nodes``'s.
    """
    cpu = collect_cpu()
    cluster = cached_cluster_digest()
    return {
        "built_at": utc_stamp(),
        "host": {
            "hostname": socket.gethostname(),
            "arch": cpu.get("arch", ""),
            "model": cpu.get("model", ""),
            "logical_cpus": cpu.get("logical_cpus"),
            "simd": cpu.get("simd", {}),
            "memory": collect_memory(),
        },
        "node_types": cluster["node_types"],
        "partitions": cluster["partitions"],
        "toolchains": collect_toolchains(),
        "signature": cluster["signature"],
        "degraded": cluster["degraded"],
    }




def collect_conda_envs() -> dict:
    """Collect available Conda environments (read-only).

    Works on typical HPC setups where conda is available as a module or binary.
    Does not activate or modify any environment.
    """
    if not cmd_exists("conda"):
        return {"available": False}

    # Use `conda env list --json` for robust parsing
    out = run(["conda", "env", "list", "--json"], timeout=10)
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

def collect_virtualenvs(workspace_root: str | None = None) -> dict:
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
