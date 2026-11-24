"""The run engine's contract: :class:`BenchTask` (one task, with ``setup`` for its
workspace), :class:`RunResult` (what a run did and how it was graded), and the adapter
that maps a benchmark onto them — a :class:`BenchmarkAdapter` grades each run in its
workspace (``score``), a :class:`BatchAdapter` takes from each workspace what its grader
needs (``collect``) and grades the runs together (``grade``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    # Python modules the scorer and the task's own code import (``numpy``): checked in
    # this interpreter, and a missing one skips the task the same way.
    modules: tuple[str, ...] = ()
    # Wall-clock limit on the agent's run; past it the run stops as Stop stops it, and the
    # workspace is graded as it stands. ``None``: no limit.
    timeout_secs: float | None = None
    # Free-form per-task data the adapter's scorer may need (e.g. expected patch).
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckContext:
    """Inputs available to a scorer: the task workspace and the live agent."""

    workspace: str          # absolute path to the task's temp workspace
    agent: Any              # the live MimirAgent (for tool-gated checks)


@dataclass
class RunResult:
    task_id: str
    passed: bool
    detail: str
    skipped: bool = False     # toolchain/prereq absent → not run, not counted in pass-rate
    tool_calls: int = 0       # tool_call events observed
    events: int = 0           # total structured events emitted (proxy for run length)
    error: str = ""           # populated when the run itself crashed
    model_calls: int = 0      # the loop's calls to the model
    prompt_tokens: int | None = None   # summed as the backend reports them; None: never reported
    nudges: dict[str, int] = field(default_factory=dict)   # reminders that reached the model
    incomplete: bool = False  # the answer itself reports the task unfinished
    elapsed_secs: float = 0.0
    variant: str = ""         # the prose variant in force ("" for the code's own)
    timed_out: bool = False   # stopped at the task's timeout_secs, graded as it stood
    pending: bool = False     # collected, awaiting a BatchAdapter's grade
    artifact: dict[str, Any] = field(default_factory=dict)   # what collect() kept

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        """The result *data* holds, as ``as_dict`` wrote it; keys of a report's own ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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


@runtime_checkable
class BatchAdapter(Protocol):
    """Maps a benchmark whose grader runs once over every answer (SWE-bench's harness).

    ``collect`` reads, while the workspace exists, what the grader needs (a patch); the
    engine keeps it in ``RunResult.artifact`` and the run stays ``pending`` until
    ``grade`` is given the pending runs together.
    """

    name: str

    def load_tasks(self, limit: int | None = None) -> list[BenchTask]:
        """Return the benchmark's tasks (capped at *limit* when given)."""
        ...

    async def collect(self, task: BenchTask, answer: str, ctx: CheckContext) -> dict[str, Any]:
        """What the grader needs of this run, JSON-serialisable."""
        ...

    async def grade(self, runs: list[RunResult], label: str
                    ) -> dict[str, tuple[bool, str] | tuple[bool, str, bool]]:
        """A verdict per ``task_id`` of *runs*; *label* names this batch (its run id)."""
        ...
