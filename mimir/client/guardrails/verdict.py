"""The model's verdict on what a run's output showed: a claim, recorded as such.

MIMIR never parses a program's output for pass/fail; the judging tool records the
model's statement about it. A verdict can lower credit, never raise it past what the
machine saw. Scopes and each verdict's reach: POLICY.md, "Verdicts".
"""
from __future__ import annotations

from typing import Any

from ..context import VERDICTS, failed_runs, unsettled_runs
from ..protocol.event_sink import emit
from .ladder import register_run_failure
from ..context.execution_context import ExecutionContext


# Verdicts that settle the one run they speak for and no other. ``pass`` because
# crediting a run the statement did not mean is the unsafe direction; ``rejected``
# because a losing candidate says nothing whatsoever about the runs around it — it
# is a statement about one measurement, and the whole reason the word exists is
# that ``fail`` was being made to carry it.
_NARROW_VERDICTS = ("pass", "rejected")


def _warn_unmatched_scope(scope: str, runs: dict[str, Any]) -> None:
    """Tell the model a ``fail``'s scope matched nothing, and what it could have named."""
    keys = ", ".join(sorted(runs)[:6]) or "none"
    emit({"type": "status", "text": (
        f"  ⚠ Verdict scope '{scope}' matched no outstanding run — applied to all "
        f"of them. Outstanding: {keys}"
    )})


def _runs_addressed(runs: dict[str, Any], scope: str) -> dict[str, Any]:
    """The outstanding runs a statement speaks for: all of them, or those its scope matches.

    Matched case-insensitively against the ledger key and the command as typed.
    """
    if not scope:
        return dict(runs)
    needle = scope.lower()
    matched = {
        key: run for key, run in runs.items()
        if needle in key.lower() or needle in str(run.get("command", "")).lower()
    }
    if matched:
        return matched
    # Either containment: the ledger key drops flags and pipes, so a precise scope
    # (`proxy_eval(op='run') OMP=64`) contains the key rather than the reverse.
    return {
        key: run for key, run in runs.items()
        if key.lower() in needle or (
            str(run.get("command", "")).lower() in needle
            if run.get("command") else False
        )
    }


def apply_verdict(
    verdict: str, reason: str, scope: str, execution_context: ExecutionContext | None,
) -> list[dict[str, Any]]:
    """Apply a stated verdict to the runs it addresses; return those it settled.

    ``fail``/``unknown``: every outstanding run. ``pass``/``rejected``: the one named, else
    the latest; ``rejected`` charges nothing. ``blocked``: failed runs. A scope matching
    nothing falls back as if absent; ``fail`` drives ``register_run_failure``.
    """
    if execution_context is None or verdict not in VERDICTS:
        return []
    if verdict == "blocked":
        return _apply_blocked(reason, scope, execution_context)
    runs = unsettled_runs(execution_context)
    if not runs:
        return []
    addressed = _runs_addressed(runs, scope)
    matched = bool(addressed)
    if not matched:
        addressed = dict(runs)
    if verdict in _NARROW_VERDICTS and (not scope or not matched):
        command = next(reversed(addressed))
        addressed = {command: addressed[command]}
    if verdict == "fail" and scope and not matched:
        _warn_unmatched_scope(scope, runs)
    settled: list[dict[str, Any]] = []
    for command, run in addressed.items():
        run["verdict"], run["reason"] = verdict, reason
        if verdict == "fail":
            register_run_failure(execution_context, command, reason)
        settled.append({**run, "command": run.get("command") or command})
    _close_exercise_advice(execution_context, verdict)
    return settled


def _apply_blocked(
    reason: str, scope: str, execution_context: ExecutionContext,
) -> list[dict[str, Any]]:
    """Re-impute failed runs to the environment, returning their repair budget.

    The run keeps ``completed=False``; only the ladder stops treating it as a defect.
    """
    runs = failed_runs(execution_context)
    if not runs:
        return []
    addressed = _runs_addressed(runs, scope) or runs
    settled: list[dict[str, Any]] = []
    for command, run in addressed.items():
        run["verdict"], run["reason"], run["blocked"] = "blocked", reason, reason
        run["failures"] = 0
        settled.append({**run, "command": run.get("command") or command})
    _close_exercise_advice(execution_context, "blocked")
    return settled


def _close_exercise_advice(execution_context: ExecutionContext, verdict: str) -> None:
    """``unknown`` and ``blocked`` close the build-it/run-it recommendation for this query."""
    if verdict in ("unknown", "blocked"):
        execution_context.execution.exercise_advice_closed = True
