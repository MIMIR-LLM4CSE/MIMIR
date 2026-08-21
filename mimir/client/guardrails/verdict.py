"""The model's verdict on what a run's output showed.

Exit 0 means a program reached its end, never that its answer is right, and no parser
generalises across the shapes real output takes — fields, convergence tables, plots,
logs, physical units. The only reader that can judge them is the model itself.

So the division of labour is deliberate and one-way: **mimir never parses the program's
output for a pass/fail; it records the model's own statement about that output.** That
statement arrives as a tool call (the ``judge`` capability), not as a line in the
assistant's prose: a structured channel keeps the bookkeeping out of what the user
reads, and leaves nothing to a grammar the model has to remember.

A verdict is about a **run**, and about nothing else. It credits no file: whether
``solver.py`` parses, imports and lints is a separate question a checker answers on its
own, and mixing the two is what let "the tests exited 0" be read back as "the answer is
right". So there is no attribution to guess here — the run is the subject.

What is recorded is a *claim*, never an observation, and the ledger renders it under its
own heading for exactly that reason. A model may lower its own credit — a ``fail`` drives
the same repair ladder a non-zero exit does — but it can never raise it past what the
machine saw.

The same asymmetry decides how far one statement reaches. ``fail`` and ``unknown``
address every outstanding run at once, because withholding credit broadly is never the
unsafe direction. A ``pass`` may only settle what it actually addresses: the run it names
through ``verdict_scope``, or — unnamed — the most recent one, which is what a model
stating a verdict right after reading an output is speaking about.

``unknown`` is the one verdict that addresses a run without closing it. The run stays
outstanding, carrying the stated verdict, because "I cannot tell" is a state somebody has
to be told about at the end. What stops that from becoming a loop is the reminder budget,
which ``unknown`` deliberately does not re-arm: the model is asked at most twice, then the
run is reported unresolved and the turn moves on.
"""
from __future__ import annotations

from typing import Any

from ..context import VERDICTS, unsettled_runs
from .observations import _register_run_failure


def _runs_addressed(runs: dict[str, Any], scope: str) -> dict[str, Any]:
    """The outstanding runs a statement speaks for.

    Unscoped, that is all of them. Scoped, it is every run whose command contains the
    named text, matched as a case-insensitive substring so the model can write the
    command or a recognisable fragment of it and be understood.

    A scope matching nothing outstanding returns nothing, and :func:`apply_verdict`
    decides what that means — the choice belongs there, where the verdict's direction
    is known.
    """
    if not scope:
        return dict(runs)
    needle = scope.lower()
    return {command: run for command, run in runs.items() if needle in command.lower()}


def apply_verdict(
    verdict: str, reason: str, scope: str, execution_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply a stated verdict to the runs it addresses; return the ones it settled.

    Each returned entry is the run's record plus its ``command``, so a caller can report
    it — the UI badge reads ``call_id`` from there. An empty list means there was no
    outstanding run to address at all.

    - ``fail`` / ``unknown`` address every outstanding run. Withholding credit from a run
      the statement did not mean costs nothing but a re-judgement.
    - ``pass`` settles the run it addresses and no other: the one it names, or — unscoped
      — the most recent one. The rest stay outstanding and are asked about on their own.

    A scope naming a run nobody can find is read as no scope at all, rather than
    discarded. Dropping it silently was the worse failure: nothing recorded, nothing
    emitted, and a reminder asking for the statement the model had just made — which a
    model answers by making it again, unchanged. The asymmetry survives the fallback,
    because it is the direction that matters: ``fail``/``unknown`` fall back to every
    run, ``pass`` to exactly one.

    ``fail`` routes through :func:`_register_run_failure`, the *same* ladder a non-zero
    exit drives — retry budget, workflow transition, and the record of what was tried.
    There is no second mechanism.
    """
    if execution_context is None or verdict not in VERDICTS:
        return []
    runs = unsettled_runs(execution_context)
    if not runs:
        return []
    addressed = _runs_addressed(runs, scope)
    matched = bool(addressed)
    if not matched:
        addressed = dict(runs)
    if verdict == "pass" and (not scope or not matched):
        command = next(reversed(addressed))
        addressed = {command: addressed[command]}
    settled: list[dict[str, Any]] = []
    for command, run in addressed.items():
        run["verdict"], run["reason"] = verdict, reason
        if verdict == "fail":
            _register_run_failure(execution_context, command, reason)
        settled.append({**run, "command": command})
    _rearm_verdict_reminders(execution_context, verdict)
    return settled


def _rearm_verdict_reminders(execution_context: dict[str, Any], verdict: str) -> None:
    """Give the reminder budget back once the model has actually judged something.

    The cap exists so two ignored reminders do not become per-step spam, but it must not
    mute the reminder for the rest of the query: a later run left unjudged deserves a
    fresh one. Mirrors the error_recovery re-arm — the budget stays spent while the
    condition persists, which an ``unknown`` verdict is (the question is still open).
    """
    if verdict == "unknown":
        return
    counts = execution_context.get("nudge_counts")
    if isinstance(counts, dict):
        counts["output_verdict"] = 0
