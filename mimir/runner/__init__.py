"""The headless run engine: the agent over a list of tasks, as a library.

A fresh agent per task, isolated workspace, every tool approved, no scoring of its own:
an integration implements :class:`BenchmarkAdapter` (graded run by run) or
:class:`BatchAdapter` (graded together) and calls :func:`run_benchmark`.
"""
from .engine import grade_pending, run_benchmark, run_one, use_backend
from .types import BatchAdapter, BenchmarkAdapter, BenchTask, CheckContext, RunResult

__all__ = [
    "run_benchmark",
    "run_one",
    "grade_pending",
    "use_backend",
    "RunResult",
    "BenchmarkAdapter",
    "BatchAdapter",
    "BenchTask",
    "CheckContext",
]
