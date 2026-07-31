"""
MCP Platform Server
===================
Builds a platform profile on demand so the agent can propose architecture-aware
scientific solutions (compiler flags, resource requests, CPU/GPU strategy). Stateless:
tools return their answer to the caller; nothing is persisted to disk.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from platform_profile_store import now_iso
from module_env import module_shell_script
from capabilities import tool_caps, ENV_DISCOVERY
from responses import err, ok

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


def _collect_cpu() -> dict:
    data = {
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if _cmd_exists("lscpu"):
        out = _run(["lscpu"])
        if out["ok"]:
            ls = _parse_lscpu(out["stdout"])
            flags = ls.get("flags", "").split()
            data.update(
                {
                    "model": ls.get("model name", ""),
                    "sockets": ls.get("socket(s)", ""),
                    "cores_per_socket": ls.get("core(s) per socket", ""),
                    "threads_per_core": ls.get("thread(s) per core", ""),
                    "numa_nodes": ls.get("numa node(s)", ""),
                    "simd": {
                        "avx2": "avx2" in flags,
                        "avx512f": "avx512f" in flags,
                        "fma": "fma" in flags,
                    },
                }
            )
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


def _collect_gpu() -> dict:
    if not _cmd_exists("nvidia-smi"):
        return {"available": False}
    query = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    out = _run_shell(query)
    if not out["ok"]:
        return {"available": False, "error": out["stderr"].strip()}
    gpus = []
    for line in out["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
    return {"available": bool(gpus), "count": len(gpus), "devices": gpus}


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


def _collect_modules() -> dict:
    script = module_shell_script("module -t avail 2>&1 | head -n 120")
    out = _run_shell(script)
    if not out["ok"]:
        return {"available": False, "error": out["stderr"].strip()}
    lines = [ln.strip() for ln in out["stdout"].splitlines() if ln.strip()]
    return {"available": True, "sample": lines[:80], "count_sample": len(lines[:80])}


def _collect_toolchains() -> dict:
    tools = {}
    for name in ["gcc", "g++", "clang", "python3", "mpicc", "mpicxx", "nvcc"]:
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



def _latest_benchmark_summary(profile: dict) -> dict:
    benches = profile.get("benchmarks", {}) if isinstance(profile, dict) else {}
    latest = benches.get("latest", {}) if isinstance(benches, dict) else {}
    return {
        "python_mops": latest.get("python_compute", {}).get("throughput_mops"),
        "memory_gbps": latest.get("memory_copy", {}).get("throughput_gbps"),
        "numpy_gflops": latest.get("numpy_matmul", {}).get("gflops"),
        "timestamp": latest.get("timestamp"),
    }


def _infer_code_traits(code_excerpt: str, language: str) -> list[str]:
    text = (code_excerpt or "").lower()
    traits = []

    if "for " in text or "while " in text:
        traits.append("iterative-kernel")
    if "numpy" in text or "np." in text:
        traits.append("numpy")
    if "mpi" in text or "mpi_" in text or "mpirun" in text:
        traits.append("mpi")
    if "openmp" in text or "#pragma omp" in text:
        traits.append("openmp")
    if "cuda" in text or "__global__" in text:
        traits.append("cuda")
    if "csr" in text or "sparse" in text:
        traits.append("sparse")
    if "fft" in text:
        traits.append("fft")
    if "float" in text and "double" not in text:
        traits.append("single-precision-like")

    if not traits:
        traits.append(f"generic-{language}")
    return sorted(set(traits))

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


@mcp.tool()
def platform_compiler_recommendations(language: str = "cpp", precision: str = "double") -> dict:
    """Recommend compiler and flags according to detected architecture.

    Args:
        language: cpp, c, fortran, python.
        precision: single or double.
    """
    profile = _build_profile()  # stateless: build fresh, never read a persisted cache
    simd = profile.get("cpu", {}).get("simd", {})
    gpu = profile.get("gpu", {}).get("available", False)

    flags = ["-O3", "-fopenmp"]
    if simd.get("avx512f"):
        flags += ["-mavx512f", "-mfma"]
    elif simd.get("avx2"):
        flags += ["-mavx2", "-mfma"]

    if precision == "single":
        flags += ["-DUSE_FLOAT32"]
    else:
        flags += ["-DUSE_FLOAT64"]

    recommendations = {
        "language": language,
        "precision": precision,
        "cpu_flags": flags,
        "gpu_available": gpu,
        "notes": [],
    }
    if gpu:
        recommendations["notes"].append("GPU detected: consider CUDA/OpenACC/OpenMP target offload.")
    if profile.get("slurm", {}).get("available"):
        recommendations["notes"].append("Slurm detected: validate scaling with single-node then multi-node jobs.")
    return ok(recommendations)


@mcp.tool()
def platform_scientific_plan(problem_type: str, scale: str = "medium", precision: str = "double") -> dict:
    """Return an architecture-aware strategy for scientific workloads.

    Args:
        problem_type: pde, linear-algebra, sparse, fft, monte-carlo, ml, other.
        scale: small, medium, large.
        precision: single or double.
    """
    profile = _build_profile()  # stateless: build fresh, never read a persisted cache
    gpu = profile.get("gpu", {}).get("available", False)
    slurm = profile.get("slurm", {}).get("available", False)

    libs = []
    strategy = []
    slurm_hints = []

    if problem_type in {"linear-algebra", "fft"}:
        libs += ["BLAS/LAPACK", "FFTW"]
        strategy += ["Start with threaded CPU baseline (OpenMP)."]
        if gpu:
            libs += ["cuBLAS", "cuFFT"]
            strategy += ["Evaluate GPU offload for large dense kernels."]
    elif problem_type in {"sparse", "pde"}:
        libs += ["PETSc", "Trilinos"]
        strategy += ["Use domain decomposition + MPI + OpenMP hybrid model."]
    elif problem_type == "monte-carlo":
        libs += ["OpenMP", "MPI"]
        strategy += ["Use embarrassingly parallel sampling with per-rank RNG streams."]
    else:
        libs += ["NumPy/SciPy", "OpenMP", "MPI"]
        strategy += ["Prototype in Python, then offload hotspots to C++ kernels."]

    if scale == "large" and slurm:
        slurm_hints += [
            "Use batch mode with checkpoints.",
            "Request resources incrementally after baseline scaling tests.",
        ]
    elif slurm:
        slurm_hints += ["Use interactive allocation first (salloc) for rapid tuning."]

    return ok({
        "problem_type": problem_type,
        "scale": scale,
        "precision": precision,
        "gpu_available": gpu,
        "slurm_available": slurm,
        "recommended_libraries": libs,
        "strategy": strategy,
        "slurm_hints": slurm_hints,
    })


@mcp.tool()
def platform_code_advisor(
    code_excerpt: str,
    language: str = "cpp",
    problem_type: str = "other",
    scale: str = "medium",
    precision: str = "double",
) -> dict:
    """Analyze a code excerpt and return platform-aware optimization advice.

    Args:
        code_excerpt: Short code snippet to inspect.
        language: cpp, c, fortran, python, cuda, other.
        problem_type: pde, linear-algebra, sparse, fft, monte-carlo, ml, other.
        scale: small, medium, large.
        precision: single or double.
    """
    if not code_excerpt.strip():
        return err("code_excerpt is empty.", hint="Provide at least a short loop/kernel or function body.")

    profile = _build_profile()  # stateless: build fresh, never read a persisted cache
    compiler = platform_compiler_recommendations(language=language, precision=precision)
    plan = platform_scientific_plan(problem_type=problem_type, scale=scale, precision=precision)
    traits = _infer_code_traits(code_excerpt, language)
    bench = _latest_benchmark_summary(profile)

    actions = []
    if "iterative-kernel" in traits and "openmp" not in traits:
        actions.append("Parallelize outer loops with OpenMP and verify thread affinity.")
    if "numpy" in traits and (bench.get("numpy_gflops") is None or bench.get("numpy_gflops") < 50):
        actions.append("Verify BLAS backend (MKL/OpenBLAS) and tune OMP_NUM_THREADS.")
    if "mpi" not in traits and scale == "large" and profile.get("slurm", {}).get("available"):
        actions.append("Plan MPI decomposition before increasing node count.")
    if "cuda" in traits and not profile.get("gpu", {}).get("available", False):
        actions.append("CUDA code detected but no GPU found on this node; keep CPU fallback path.")
    if "sparse" in traits:
        actions.append("Use sparse-native kernels and reduce indirect memory access where possible.")
    if "fft" in traits:
        actions.append("Prefer batched FFT plans and reuse plans across timesteps.")
    if precision == "single" and "single-precision-like" not in traits:
        actions.append("If numerically stable, consider float32 for memory-bandwidth-bound kernels.")

    if not actions:
        actions.append("Establish baseline runtime, then profile and optimize top hotspots only.")

    return ok({
        "language": language,
        "problem_type": problem_type,
        "scale": scale,
        "precision": precision,
        "detected_traits": traits,
        "platform_snapshot": {
            "cpu": profile.get("cpu", {}),
            "gpu": profile.get("gpu", {}),
            "slurm": profile.get("slurm", {}),
        },
        "benchmark_snapshot": bench,
        "compiler_recommendations": compiler,
        "scientific_plan": plan,
        "action_items": actions,
        "next_step": "Run benchmark_summary(persist_to_platform_profile=True), then re-run platform_code_advisor after code changes.",
    })


if __name__ == "__main__":
    mcp.run()
