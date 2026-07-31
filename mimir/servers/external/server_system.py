"""
MCP System Server
=================
Read-only tools that expose OS metrics and environment information.
No shell execution; all data comes from the Python standard library.
"""

import os
import platform
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok
from capabilities import tool_caps

mcp = FastMCP(
    "SystemServer",
    debug=False,
    log_level="ERROR",
)
_WORKSPACE_ROOT = os.path.abspath(os.environ.get("MCP_FILES_ROOT", "/"))

# A fixed allowlist of environment variables the agent may read.
_SAFE_ENV_VARS = {
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ",
    "PATH", "HOSTNAME", "PWD", "TERM", "EDITOR",
    # ML / HPC environment
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "SLURM_JOB_ID", "SLURM_NODELIST",
}

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── tool — single read-only dispatch ──────────────────────────────────────────

_SYSTEM_OPS = ("info", "disk", "cpu", "memory", "env", "uptime")


def _op_info() -> dict:
    return ok({
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    })


def _op_disk(path: str) -> dict:
    full = os.path.abspath(path)
    if _WORKSPACE_ROOT != "/" and not full.startswith(_WORKSPACE_ROOT + os.sep) and full != _WORKSPACE_ROOT:
        return err(
            f"Path '{path}' is outside the workspace root.",
            hint="Use a path inside the workspace or '/' when MCP_FILES_ROOT is not set.",
        )
    try:
        total, used, free = shutil.disk_usage(path)
        return ok({
            "path": path,
            "total_gb": round(total / 1e9, 2),
            "used_gb":  round(used  / 1e9, 2),
            "free_gb":  round(free  / 1e9, 2),
            "used_pct": round(used / total * 100, 1),
        })
    except Exception as e:
        return err(str(e), hint="Check that the path exists and is readable.")


def _op_cpu() -> dict:
    result: dict = {"cpu_count": os.cpu_count()}
    try:
        result["load_avg_1m"], result["load_avg_5m"], result["load_avg_15m"] = (
            round(x, 2) for x in os.getloadavg()
        )
    except OSError:
        pass
    if _HAS_PSUTIL:
        result["cpu_percent"] = psutil.cpu_percent(interval=0.5)
    return ok(result)


def _op_memory() -> dict:
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        return ok({
            "total_gb":     round(vm.total     / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "used_gb":      round(vm.used      / 1e9, 2),
            "used_pct":     vm.percent,
        })
    # Fallback: parse /proc/meminfo (Linux)
    try:
        info: dict = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if parts[0].rstrip(":") in ("MemTotal", "MemFree", "MemAvailable"):
                    info[parts[0].rstrip(":")] = int(parts[1])  # kB
        return ok({
            "total_gb":     round(info.get("MemTotal",     0) / 1e6, 2),
            "available_gb": round(info.get("MemAvailable", 0) / 1e6, 2),
            "used_gb":      round((info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / 1e6, 2),
        })
    except Exception as e:
        return err(str(e), hint="Memory stats are unavailable on this host.")


def _op_env(name: str) -> dict:
    if name not in _SAFE_ENV_VARS:
        return err(
            f"'{name}' is not in the allowed variable list.",
            hint=f"Allowed variables: {sorted(_SAFE_ENV_VARS)}",
        )
    return ok({"name": name, "value": os.environ.get(name, "(not set)")})


def _op_uptime() -> dict:
    now = time.time()
    result: dict = {"epoch": int(now)}
    if _HAS_PSUTIL:
        boot = psutil.boot_time()
        secs = int(now - boot)
        result["uptime_seconds"] = secs
        result["uptime_human"] = f"{secs // 3600}h {(secs % 3600) // 60}m"
    return ok(result)


@mcp.tool(**tool_caps(label="System {op}"))
def system(op: str, path: str = "/", name: str = "") -> dict:
    """Read-only OS metrics and environment info, selected by ``op``.

    Operations (set ``op`` to one of these):
      info   -> OS + Python runtime (os, architecture, hostname, cpu_count, ...)
      disk   -> disk usage for `path` (total/used/free GB, used_pct)
      cpu    -> CPU count + load averages (+ cpu_percent if psutil present)
      memory -> RAM usage (psutil, else /proc/meminfo fallback)
      env    -> value of allow-listed env var `name`
      uptime -> system uptime + current epoch

    Args:
        op:   The operation to perform (see list above).
        path: Filesystem path for ``disk`` (default '/'). Workspace-scoped.
        name: Env var name for ``env`` (only a safe allowlist is accessible).
    """
    if op == "info":
        return _op_info()
    if op == "disk":
        return _op_disk(path)
    if op == "cpu":
        return _op_cpu()
    if op == "memory":
        return _op_memory()
    if op == "env":
        return _op_env(name)
    if op == "uptime":
        return _op_uptime()
    return err(
        f"Unknown system op '{op}'.",
        hint=f"Use one of: {', '.join(_SYSTEM_OPS)}.",
    )


if __name__ == "__main__":
    mcp.run()
