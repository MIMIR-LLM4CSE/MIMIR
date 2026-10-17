"""Whether the change was ever run, and the cheapest way to run it.

The advisory axis. ``regression`` and ``unexercised`` are two phrasings of one question,
"does anything show this works?", so they ration one budget (``EXERCISE_BUDGET``), and
asking it is a recommendation: a build needs a toolchain and a run an environment, which
the loop cannot conjure. Once the model has answered (a run it judged, or ``unknown``
saying it cannot), the subject is closed for the rest of the query.

No nudge asks for a *verdict* on a run: the VERDICT_DUE annotation on the run's own
result and the judging tool's docstring ask for it where it costs nothing.
"""
from __future__ import annotations

from __future__ import annotations
import os
from ..workflow import (
    has_pending_validation,
)
from ...context.execution_context import (
    known_existing_files,
    nudge_count,
)
from ....servers._shared.shell_paths import any_command_on_path
from ...context.capabilities import CODE_EXEC, names_with_cap
from ...config.limits import (
    EXERCISE_BUDGET,
    NUDGE_MAX_EXERCISE,
)
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from .predicates import untested_edited_sources


# What this box can start with one direct command, by edited-file kind. The value is the
# runner that has to exist for that to be true — asked of PATH, never assumed, so a `.py`
# edit on a box with no interpreter is correctly not a route.
_DIRECT_RUNNERS: dict[str, tuple[str, ...]] = {
    ".py": ("python3", "python"),
    ".sh": ("sh",),
}


# A suite already registered against a configured tree: `ctest` is one direct command, and
# unlike a build it produces a result somebody can judge. The file is generated at configure
# time and only when tests exist, so its presence is the whole test — a compiled test
# *source* is deliberately not a route, because reaching it still means building first.
_REGISTERED_SUITES: dict[str, tuple[str, ...]] = {
    "CTestTestfile.cmake": ("ctest",),
}


# A build only counts as a route when the tree is ALREADY configured — that is the whole
# proportionality test. `CMakeCache.txt` and not `CMakeLists.txt`: a configured tree is one
# direct command, configuring one is a step of its own and stays out of reach.
_CONFIGURED_BUILDS: dict[str, tuple[str, ...]] = {
    "Makefile": ("make",),
    "makefile": ("make",),
    "GNUmakefile": ("make",),
    "CMakeCache.txt": ("cmake", "make"),
}


def _exercise_budget_left(execution_context: ExecutionContext) -> bool:
    """True while the one shared exercise/verdict ask is still available."""
    return (
        not execution_context.exercise_advice_closed
        and nudge_count(execution_context, EXERCISE_BUDGET) < NUDGE_MAX_EXERCISE
    )


def exercise_route(agent: AgentPort, execution_context: ExecutionContext) -> tuple[str, str]:
    """``(route, obstacle)``: the one direct command that would exercise this change,
    named — or ``""`` and what stood in the way.

    This is where "recommended when it is simply feasible" is decided, rather than
    recited. A route is one invocation against the project as it stands, using something
    that is actually installed; anything needing a step of its own first — configuring a
    build, installing a package, provisioning data — is out of proportion and returns "".
    So the model is never pushed toward a disproportionate step, and a recommendation that
    does fire can name the command it found.

    Silent, but not invisible: the obstacle is what :func:`record_exercise_obstacle` puts
    in the ledger instead of simply omitting the advice. A pure function — it is asked from
    a nudge predicate, and probing a predicate must change nothing. Evidence only — a build file has to have
    been *seen* this session, because recommending `make` against a Makefile nobody has laid
    eyes on is not obviously proportionate either.
    """
    def _blocked(reason: str) -> tuple[str, str]:
        return "", reason

    def _route(description: str) -> tuple[str, str]:
        return description, ""

    if execution_context.unresolved_modules:
        return _blocked("an import could not be resolved in the interpreter that was used")
    if not names_with_cap(CODE_EXEC, getattr(agent, "tool_caps", None)):
        return _blocked("no execution tool is connected to this session")

    # 1. A test that already covers the edit: the cheapest route there is, and the one the
    #    regression row is built on.
    untested = untested_edited_sources(execution_context)
    if untested:
        return _route(f"the test that already covers it ({untested[0][1]})")

    dirty = sorted(execution_context.dirty_written_files or ())
    known = {os.path.basename(p) for p in known_existing_files(execution_context)}

    # 2. A file this environment starts directly.
    for path in dirty:
        runners = _DIRECT_RUNNERS.get(os.path.splitext(path)[1].lower())
        if runners and any_command_on_path(runners):
            return _route(f"running {path} directly")

    needs_build = [p for p in dirty if os.path.splitext(p)[1].lower() not in _DIRECT_RUNNERS]
    if not needs_build:
        return _blocked("nothing written here starts with one direct command")

    # 3. A suite already registered here — ahead of the plain build below, and for the
    #    reason the tier ladder gives: a build says the code is well formed, only a run
    #    produces a result. This is what route 1 is for a compiled language, where pairing
    #    a test *file* to a source would name something that still has to be built.
    for suite_file, runners in _REGISTERED_SUITES.items():
        if suite_file in known and any_command_on_path(runners):
            return _route(f"the test suite already registered here ({runners[0]})")

    # 4. A build already configured in the tree: the route for a compiled change with
    #    no registered suite.
    for build_file, drivers in _CONFIGURED_BUILDS.items():
        if build_file in known and any_command_on_path(drivers):
            return _route(f"the build already configured here ({build_file})")

    return _blocked("nothing written here starts with one direct command")


def exercise_hint(agent: AgentPort, execution_context: ExecutionContext) -> str:
    return exercise_route(agent, execution_context)[0]


def record_exercise_obstacle(agent: AgentPort, execution_context: ExecutionContext) -> None:
    """Note on the context why the change could not be exercised, for the ledger.

    Asked where the ledger is built, for the case the ledger reports it in: code was
    written and nothing was run.
    """
    if execution_context.dirty_written_files and not execution_context.runs:
        execution_context.exercise_blocked_reason = exercise_route(
            agent, execution_context)[1]


def should_nudge_regression(execution_context: ExecutionContext) -> bool:
    """Budget left and an edited source has an on-disk test that was never run."""
    return (
        _exercise_budget_left(execution_context)
        and bool(untested_edited_sources(execution_context))
    )


def should_nudge_unexercised_code(agent: AgentPort, execution_context: ExecutionContext) -> bool:
    """Shared budget left, every written file checked, and nothing was ever run.

    Mutually exclusive with the two rows around it: ``validation`` speaks while a file
    still owes a check, this one owns the state after — checks all green, no run at all
    — which is exactly where a model stops believing it is finished. A checker
    established that the file is written correctly; nothing yet bears on whether it
    computes the right thing, and nothing will until something executes.

    A recommendation, not a requirement, which is why it is bounded three ways: the
    shared exercise budget, the feasibility gate, and the fact that "there was nothing
    to run" is an answer the message explicitly accepts. Nothing downstream blocks on
    it — an unexercised change concludes, and the ledger says so.
    """
    return (
        _exercise_budget_left(execution_context)
        and bool(execution_context.code_mutation_started)
        and bool(execution_context.dirty_written_files)
        and not has_pending_validation(execution_context)
        and not (execution_context.runs or {})
        and bool(exercise_hint(agent, execution_context))
    )
