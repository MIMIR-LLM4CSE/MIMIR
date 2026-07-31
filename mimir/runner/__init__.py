"""Headless run engine for the MCP agent — the agent's "batch mode".

Drives the agent non-interactively over a list of tasks (fresh agent per task,
isolated workspace, every tool auto-approved, structured-event capture, missing
toolchains skipped) and returns a JSON-serialisable summary. It does **no
scoring** — a benchmark supplies tasks *and* grades the result via the
:class:`BenchmarkAdapter` contract.

This package is a **library**, not a CLI: an external integration imports it,
implements :class:`BenchmarkAdapter` over its benchmark, and calls
:func:`run_benchmark`. The agent therefore depends on no benchmark, and a
benchmark depends on no agent — only the integration knows both.

    from mimir.runner import run_benchmark, BenchTask, BenchmarkAdapter, CheckContext
"""
from .engine import RunResult, run_benchmark, run_one
from .types import BenchmarkAdapter, BenchTask, CheckContext

__all__ = [
    "run_benchmark",
    "run_one",
    "RunResult",
    "BenchmarkAdapter",
    "BenchTask",
    "CheckContext",
]
