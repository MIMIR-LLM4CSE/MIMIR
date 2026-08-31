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
``solver.py`` parses and is whole is a separate question the built-in check answers on
its own, and mixing the two is what let "the tests exited 0" be read back as "the answer is
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
to be told about at the end. Nothing loops on that: no reminder asks for a verdict, so an
outstanding run is simply reported unresolved in the ledger and the turn moves on.

``blocked`` is the only one that speaks about a run the machine already judged red, and it
does not argue with that judgement — it re-imputes it. An exit code says *that* a run
failed, never *whose* fault it was, and a wall the environment put there is not a defect in
the change: it costs no repair budget and is reported as a limitation rather than as an
unfinished task. The run stays as red as the machine saw it, so nothing is raised past that;
what changes is only who is charged. And it is a retraction the model must claim: an
unclaimed red exit drives the repair ladder exactly as before.
"""
from __future__ import annotations

from typing import Any

from ..context import VERDICTS, failed_runs, unsettled_runs
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
    - ``blocked`` addresses *failed* runs instead, which is a disjoint set: a run that never
      completed is not outstanding, it is already judged. See :func:`_apply_blocked`.

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
    if verdict == "blocked":
        return _apply_blocked(reason, scope, execution_context)
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
    _close_exercise_advice(execution_context, verdict)
    return settled


def _apply_blocked(
    reason: str, scope: str, execution_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Re-impute failed runs from the change to the environment; return the ones addressed.

    Scoped like ``fail``: unscoped it speaks for every failed run, because a wall this box
    put there is rarely specific to one command, and a scope naming nothing falls back to
    all of them rather than being dropped.

    Returning the repair budget is the whole effect. The run keeps ``completed=False`` — the
    machine saw a red exit and that stands — so no caller can read this as a success; what
    stops is only the ladder that was treating it as a defect to fix.
    """
    runs = failed_runs(execution_context)
    if not runs:
        return []
    addressed = _runs_addressed(runs, scope) or runs
    settled: list[dict[str, Any]] = []
    for command, run in addressed.items():
        run["verdict"], run["reason"], run["blocked"] = "blocked", reason, reason
        run["failures"] = 0
        settled.append({**run, "command": command})
    _close_exercise_advice(execution_context, "blocked")
    return settled


def _close_exercise_advice(execution_context: dict[str, Any], verdict: str) -> None:
    """``unknown`` and ``blocked`` end the build-it/run-it recommendation for this query.

    Both are accepted answers to the whole advisory question — one says the output cannot
    be read, the other that the environment will not produce one — and asking again after
    either is asking for a different answer to a question already answered.

    The other verdicts do nothing here. There used to be a symmetric re-arm, handing the
    shared exercise budget back once nothing was left outstanding, because a reminder
    asked for the verdict itself; that reminder is gone (see the advisory-axis comment
    in nudges/engine.py) and re-arming a *run* recommendation on the strength of a
    verdict would only ask a model that just judged its run to go and run more.
    """
    if verdict in ("unknown", "blocked"):
        execution_context["exercise_advice_closed"] = True
