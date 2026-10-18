"""The headless run engine: the agent over a list of tasks, as a library.

A fresh agent per task, isolated workspace, every tool approved, no scoring of its own:
an integration implements :class:`BenchmarkAdapter` and calls :func:`run_benchmark`.
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
