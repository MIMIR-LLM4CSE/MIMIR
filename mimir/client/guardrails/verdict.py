"""The model's verdict on what a run's output showed.

Exit 0 means a program reached its end, never that its answer is right, and no parser
generalises across the shapes real output takes — fields, convergence tables, plots,
logs, physical units. The only reader that can judge them is the model itself.

So the division of labour is deliberate and one-way: **mimir never parses the program's
output for a pass/fail; it parses the model's own statement about that output.** Program
output is unbounded and belongs to whoever wrote the code; the verdict line is a format
the model controls, which is the only place a fixed grammar is legitimate.

What is recorded is a *claim*, never an observation, and the ledger renders it under its
own label for exactly that reason. A model may lower its own credit — a ``fail`` drives
the same repair ladder a non-zero exit does — but it can never raise it past what the
machine saw: a self-declared failure does not arm the red→green promotion.

The same asymmetry decides how far one statement reaches. ``fail`` and ``unknown``
settle every outstanding run at once, because withholding credit broadly is never the
unsafe direction. A ``pass`` may only credit what it actually addresses: when several
runs are outstanding and they bear on *different* files, it must name its run —
``verdict[solver.py]: pass — …`` — or it settles nothing and is asked to say which.
"""
from __future__ import annotations

import re
from typing import Any

from ..context import VERDICTS
from .observations import (
    _mark_file_validated,
    _register_validation_failure,
    record_output_verdict,
)

#: ``verdict: pass — the L2 error is 3e-4 against the analytic solution``.
#: Tolerant about what the model puts around it (a bullet, bold, any dash or colon as
#: the separator, any case) and strict about the shape, so prose mentioning the word
#: does not register. The reason is optional in the grammar but asked for in the prompt;
#: the ``verdict[…]`` bracket naming which run is judged is optional in both.
_VERDICT_RE = re.compile(
    r"^[\s>*\-]*\**\s*verdict\s*"
    r"(?:\[(?P<scope>[^\]\n]+)\])?"
    r"\s*\**\s*[:=]\s*\**\s*"
    r"(?P<verdict>pass|fail|unknown)\b\**"
    r"(?:\s*[—–\-:]\s*(?P<reason>.+?))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_verdict(content: str) -> tuple[str, str, str] | None:
    """The ``(verdict, reason, scope)`` stated in *content*, or None if it states none.

    *scope* is the optional ``verdict[…]`` bracket naming which run is being judged —
    a command or a path, matched as a substring — and is ``""`` when the statement is
    unscoped.

    The **last** match wins: a model that reasons its way to a different conclusion
    later in the same message meant the later one, and taking the first would freeze an
    opinion it went on to revise.
    """
    if not content:
        return None
    last = None
    for m in _VERDICT_RE.finditer(content):
        last = m
    if last is None:
        return None
    verdict = last.group("verdict").lower()
    if verdict not in VERDICTS:
        return None
    return verdict, (last.group("reason") or "").strip(), (last.group("scope") or "").strip()


def _runs_addressed(runs: dict[str, Any], scope: str) -> dict[str, Any]:
    """The outstanding runs a statement speaks for.

    Unscoped, that is all of them. Scoped, it is every run whose command or one of whose
    paths contains the bracketed text, matched as a case-insensitive substring so the
    model can write the command, the file, or a recognisable fragment of either and be
    understood. A scope matching nothing outstanding addresses nothing — better than
    guessing which run was meant.
    """
    if not scope:
        return dict(runs)
    needle = scope.lower()
    return {
        command: run for command, run in runs.items()
        if needle in command.lower()
        or any(needle in path.lower() for path in run.get("paths") or ())
    }


def _bears_on_different_files(runs: dict[str, Any]) -> bool:
    """True when *runs* credit more than one distinct set of files.

    Two commands attributed to the same pending file leave nothing to disambiguate —
    one statement covers both however it is phrased. It is runs bearing on *different*
    files that make an unscoped ``pass`` a claim about work it may never have read.

    A path-less run (a scratchpad probe, an analysis-only session) is not counted: it
    credits nothing, so settling it alongside another cannot over-credit anything, and
    counting it would demand a scope on the commonest shape there is — one probe and
    one real run.
    """
    crediting = {frozenset(p) for p in (run.get("paths") or () for run in runs.values()) if p}
    return len(crediting) > 1


def record_verdict(content: str, execution_context: dict[str, Any] | None) -> bool:
    """Apply the verdict stated in an assistant message to the runs it addresses.

    Returns True when a verdict was found and applied. Which runs it settles follows
    the same asymmetry that governs the verdict itself — a model may lower its own
    credit broadly, and may only raise it precisely:

    - ``fail`` / ``unknown`` settle every outstanding run. Withholding credit from a run
      the statement did not mean costs nothing but a re-judgement.
    - ``pass`` settles every outstanding run too *when they all bear on the same files*
      — the model saw those outputs and spoke after them, and there is nothing to
      disambiguate. When they bear on different files, an unscoped ``pass`` would credit
      files the statement may never have been about, so it settles nothing and records
      what the model must choose between (``verdict_scope_required``, which the nudge
      reads back to it). Naming the run — ``verdict[solver.py]: pass — …`` — settles
      exactly that one, and the rest stay pending.

    Routing goes through the *existing* entry points on purpose — ``pass`` through
    :func:`_mark_file_validated` (so the red→green promotion, the fail-count reset and
    the conclude transition happen exactly as for any other validation) and ``fail``
    through :func:`_register_validation_failure` (so the retry budget, the anti-thrashing
    guard and the incomplete-finalization report all behave as they do for a non-zero
    exit). There is no second ladder. ``unknown`` settles the run but validates nothing:
    the file stays pending, and saying so is an acceptable ending — staying silent is not.
    """
    if execution_context is None:
        return False
    parsed = parse_verdict(content)
    if parsed is None:
        return False
    verdict, reason, scope = parsed
    runs = execution_context.get("unjudged_runs") or {}
    if not runs:
        return False
    addressed = _runs_addressed(runs, scope)
    if not addressed:
        return False
    if verdict == "pass" and not scope and _bears_on_different_files(addressed):
        execution_context["verdict_scope_required"] = sorted(addressed)
        return False
    execution_context["verdict_scope_required"] = []
    for command, run in addressed.items():
        tier = run.get("tier") or "executed"
        for path in run.get("paths") or ():
            record_output_verdict(execution_context, path, verdict, reason, command)
            if verdict == "pass":
                _mark_file_validated(execution_context, path, tier)
            elif verdict == "fail":
                _register_validation_failure(
                    execution_context, path, tier, arms_red_green=False,
                )
        runs.pop(command, None)
    execution_context["unjudged_runs"] = runs
    _rearm_verdict_reminders(execution_context, verdict)
    return True


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
