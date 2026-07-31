"""Core types for the run engine — the contract an integration implements.

A :class:`BenchTask` is one unit of work for the agent. An adapter (see
:class:`BenchmarkAdapter`) turns a benchmark into a stream of these tasks and
grades the agent's answer for each. The two seams that let *any* benchmark plug in:

- ``BenchTask.setup`` — materialise the task's workspace (seed files for a
  synthetic suite, or a repo checkout for SWE-bench-style benchmarks). It runs
  *before* the MCP file/search servers are spawned, so they root at the
  populated workspace.
- ``BenchmarkAdapter.score`` — grade the run by inspecting the workspace; an
  external benchmark calls its own evaluator (e.g. compiles/runs the solution or
  runs the instance's test suite) inside ``ctx.workspace``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class BenchTask:
    """One benchmark item fed to the agent in an isolated workspace."""

    id: str
    query: str
    mode: str = "agent"
    max_steps: int = 30
    # ``None`` → connect every server in ``config.SERVERS``. A tuple/list limits
    # the run to those servers (faster, fewer tools, closer to the old harness).
    servers: tuple[str, ...] | None = ("files", "search", "bash")
    # Populate the (already-created, empty) workspace dir. Receives its abs path.
    setup: Callable[[str], None] | None = None
    # Executables that must be on PATH to evaluate this task; if any is missing the
    # harness records the task as *skipped* (not failed) without running the agent.
    requires: tuple[str, ...] = ()
    # Free-form per-task data the adapter's scorer may need (e.g. expected patch).
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckContext:
    """Inputs available to a scorer: the task workspace and the live agent."""

    workspace: str          # absolute path to the task's temp workspace
    agent: Any              # the live MimirAgent (for tool-gated checks)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Maps a benchmark onto the run engine.

    Implemented by an external integration package (which imports both
    ``mimir.runner`` and the benchmark) and passed to :func:`run_benchmark`.
    The agent ships no adapters of its own.
    """

    name: str

    def load_tasks(self, limit: int | None = None) -> list[BenchTask]:
        """Return the benchmark's tasks (capped at *limit* when given)."""
        ...

    async def score(
        self, task: BenchTask, answer: str, ctx: CheckContext
    ) -> tuple[bool, str] | tuple[bool, str, bool]:
        """Grade *answer*; return ``(passed, human-readable detail)``.

        A scorer that provides a prerequisite itself (e.g. loads a toolchain) and
        finds it unavailable may return a third element ``skipped=True`` to record
        the task as *skipped* (excluded from pass-rate) rather than failed.
        """
        ...
