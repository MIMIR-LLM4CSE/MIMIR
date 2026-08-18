"""
MCP Benchmark Server
====================
Lightweight micro-benchmarks to calibrate architecture-aware recommendations.

These are not replacement-grade HPC benchmarks, but quick probes that can guide
initial strategy choices before running full production benchmarks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

import numpy as np
import time
from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, CODE_EXEC, PLAN_BLOCKED, RECOVERABLE, REVERSIBLE
from platform_profile_store import now_iso, persist_benchmark_report
from responses import err, ok

mcp = FastMCP(
    "BenchmarkServer",
    debug=False,
    log_level="ERROR",
)


# `reversibility` is stated rather than derived: CODE_EXEC alone would infer RECOVERABLE
# and gate an in-process micro-benchmark behind an approval prompt.
@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CODE_EXEC], reversibility=REVERSIBLE))
def benchmark_python_compute(n: int = 2_000_000, repeats: int = 3) -> dict:
    """Estimate scalar compute throughput using a pure-Python math loop.

    Args:
        n: Number of loop iterations per repeat.
        repeats: Number of repeats.
    """
    if n < 10_000 or repeats < 1:
        return err("n must be >= 10000 and repeats >= 1")

    timings = []
    checksum = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        acc = 0.0
        for i in range(1, n + 1):
            x = i * 1e-6
            acc += np.sin(x) * np.cos(x)
        t1 = time.perf_counter()
        checksum += acc
        timings.append(t1 - t0)

    best = min(timings)
    mops = (n / best) / 1e6
    return ok(
        {
            "benchmark": "python_compute",
            "n": n,
            "repeats": repeats,
            "best_time_s": round(best, 6),
            "mean_time_s": round(sum(timings) / len(timings), 6),
            "throughput_mops": round(mops, 2),
            "checksum": round(checksum, 6),
        }
    )


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CODE_EXEC], reversibility=REVERSIBLE))
def benchmark_memory_copy(size_mb: int = 256, repeats: int = 3) -> dict:
    """Estimate memory copy throughput with bytearray copies.

    Args:
        size_mb: Buffer size in MB.
        repeats: Number of measured copies.
    """
    if size_mb < 16 or repeats < 1:
        return err("size_mb must be >= 16 and repeats >= 1")

    size_bytes = size_mb * 1024 * 1024
    src = bytearray(size_bytes)
    dst = bytearray(size_bytes)

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst[:] = src
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    best = min(timings)
    gbps = (size_bytes / best) / 1e9
    return ok(
        {
            "benchmark": "memory_copy",
            "size_mb": size_mb,
            "repeats": repeats,
            "best_time_s": round(best, 6),
            "mean_time_s": round(sum(timings) / len(timings), 6),
            "throughput_gbps": round(gbps, 3),
        }
    )


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CODE_EXEC], reversibility=REVERSIBLE))
def benchmark_numpy_matmul(n: int = 1024, repeats: int = 3) -> dict:
    """Estimate dense linear algebra performance with NumPy matmul.

    Args:
        n: Matrix size (n x n).
        repeats: Number of repeats.
    """
    try:
        import numpy as np
    except Exception:
        return err("NumPy is not available.", hint="Install numpy in your active environment.")

    if n < 128 or repeats < 1:
        return err("n must be >= 128 and repeats >= 1")

    a = np.random.rand(n, n).astype(np.float64)
    b = np.random.rand(n, n).astype(np.float64)

    # Warm-up to avoid first-call overhead bias.
    _ = a @ b

    timings = []
    csum = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = a @ b
        t1 = time.perf_counter()
        timings.append(t1 - t0)
        csum += float(c[0, 0])

    best = min(timings)
    gflops = (2.0 * (n ** 3) / best) / 1e9
    return ok(
        {
            "benchmark": "numpy_matmul",
            "n": n,
            "repeats": repeats,
            "dtype": "float64",
            "best_time_s": round(best, 6),
            "mean_time_s": round(sum(timings) / len(timings), 6),
            "gflops": round(gflops, 2),
            "checksum": round(csum, 6),
        }
    )


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="can update persisted platform benchmark history",
))
def benchmark_summary(persist_to_platform_profile: bool = True) -> dict:
    """Run a compact benchmark suite and return a consolidated report.

    Args:
        persist_to_platform_profile: If true, save benchmark results into
            server_platform's `platform_profile.json` under `benchmarks`.
    """
    py = benchmark_python_compute(n=1_000_000, repeats=2)
    mem = benchmark_memory_copy(size_mb=128, repeats=2)
    mm = benchmark_numpy_matmul(n=512, repeats=2)
    report = {
        "timestamp": now_iso(),
        "python_compute": py,
        "memory_copy": mem,
        "numpy_matmul": mm,
    }

    response = ok(report)
    if persist_to_platform_profile:
        response["profile_sync"] = persist_benchmark_report(report)
    return response


if __name__ == "__main__":
    mcp.run()
