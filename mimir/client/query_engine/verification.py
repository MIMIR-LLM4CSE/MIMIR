"""The verification ledger: what a run actually established, machine-recorded.

Built from the execution context on every exit path and appended to the answer, so
history — and the model's next turn — carries the evidence independently of whatever
prose the model chose. The block opens with a marker line carrying its status and a
one-line summary: front-ends split on it (:func:`split_answer_ledger`) and render a
collapsed panel instead of trailing prose, and nothing is lost if they don't.

Report-only, emitted after the model has stopped acting: it cannot loop, cannot be
argued with, and blocks nothing.
"""
from __future__ import annotations

import re

from ..context import (
    unwritten_declared_files,
    validation_tier,
    weakest_validation_tier,
)
from ..guardrails.workflow import unchecked_checklist_items

#: Opening marker of a rendered ledger block. Stable — UIs split answers on it.
LEDGER_MARKER = "<!--mimir:ledger"

_MARKER_RE = re.compile(r"<!--mimir:ledger(?P<attrs>[^>]*)-->")
_ATTR_RE = re.compile(r'(?P<key>\w+)="(?P<value>[^"]*)"')

#: Kept as the block's first prose line so the framing reaches the model even when
#: a front-end drops the marker comment.
LEDGER_FRAMING = "Verification ledger — machine-recorded, not model-authored:"

# The absence line is domain-neutral on purpose: worded numerically it fires on every
# parser/CLI/refactor run that could never satisfy it, and becomes wallpaper.
# It only ever fires when nothing ran, so it must not point at run rows: there are none
# below it, and sending the reader to an empty half of the ledger is how a caveat stops
# being read at all.
_UNCHECKED_OUTPUT_NOTE = (
    "A checker says a file parses, imports and lints. It says nothing about whether the "
    "answer is right, and nothing here was built or run — so no result was produced, and "
    "none was judged."
)
# Not a gap the model can close: nothing on this machine can check these files at all.
_UNVERIFIABLE_NOTE = (
    "No checker for these files exists in this environment, so the check that is "
    "otherwise required could not be run on them."
)
# Said next to _UNCHECKED_OUTPUT_NOTE, never alone: "nothing ran" is the finding, this
# is why. Recorded by the feasibility gate that suppressed the run recommendation, so
# the reader learns what stood in the way instead of seeing the advice quietly absent.
_EXERCISE_BLOCKED_NOTE = "Running it was out of reach here: {reason}."
# Said once, next to the rows that carry a verdict: the ledger is machine-recorded, but
# a verdict is the one line in it the model wrote, and the two must not read alike.
_VERDICT_NOTE = (
    "Verdicts are the model's own reading of a run's output, recorded as stated and "
    "never checked — exit 0 says a program ended, not that its answer is right."
)


def _run_row(command: str, run: dict) -> str:
    """One execution, as what is actually known about it."""
    reason = str(run.get("reason") or "").strip()
    tail = f" — {reason}" if reason else ""
    # Before the not-completed branch, which a blocked run also satisfies: what matters
    # about it is not that it stopped, but that it was never really attempted here.
    if run.get("blocked"):
        return f"`{command}` — **not attempted**: {run['blocked']}"
    if not run.get("completed"):
        return f"`{command}` — **did not complete**{tail}"
    verdict = run.get("verdict")
    if verdict == "pass":
        return f"`{command}` — ran; verdict: pass{tail}"
    if verdict == "fail":
        return f"`{command}` — ran; **verdict: fail**{tail}"
    if verdict == "unknown":
        return f"`{command}` — ran; **judged unknown**{tail}"
    return f"`{command}` — ran; **its output was never judged**"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def build_ledger(execution_context: dict) -> dict | None:
    """The ledger for this run, or None when nothing happened worth recording.

    Returns ``{"status": ok|note|warn, "files": int, "summary": str, "rows": [markdown]}``.
    ``status`` separates a clean run from soft caveats and from hard gaps (unvalidated or
    unwritten files, unjudged runs, open checklist steps) — it is what a front-end colours
    the panel by.

    Two kinds of row, kept apart on purpose. A **file** row says what a checker
    established: it parses, it lints, nothing more. A **run** row says what happened when
    the code was executed and what the model read in the output. Merging them is what let
    "validated" be reported as "correct".
    """
    written = sorted(execution_context.get("dirty_written_files", set()))
    validated = execution_context.get("validated_files", set())
    unchecked = unchecked_checklist_items(execution_context)
    unwritten = unwritten_declared_files(execution_context)
    runs = execution_context.get("runs") or {}
    required = [it for it in unchecked if not it.get("optional")]
    optional = [it for it in unchecked if it.get("optional")]

    # A run nobody judged is worth reporting even when no file was touched: an
    # analysis-only session is exactly the one whose whole answer rests on that output.
    if not written and not unchecked and not unwritten and not runs:
        return None

    rows: list[str] = []
    notes: list[str] = []
    unverifiable = set(execution_context.get("unverifiable_files", set()) or set())
    unvalidated = [f for f in written if f not in validated]

    for f in written:
        if f in unvalidated:
            rows.append(
                f"`{f}` — **not checked** (no checker here)" if f in unverifiable
                else f"`{f}` — **not checked**"
            )
        else:
            rows.append(f"`{f}` — checked: {validation_tier(execution_context, f) or 'static'}")

    for command, run in sorted(runs.items()):
        rows.append(_run_row(command, run))

    open_runs = [c for c, r in runs.items() if r.get("completed") and r.get("verdict") in ("", "unknown")]
    failed = [
        c for c, r in runs.items()
        if (not r.get("completed") or r.get("verdict") == "fail") and not r.get("blocked")
    ]
    blocked = [c for c, r in runs.items() if r.get("blocked")]
    if written and not unvalidated and not runs:
        notes.append(_UNCHECKED_OUTPUT_NOTE)
        blocked = str(execution_context.get("exercise_blocked_reason") or "").strip()
        if blocked:
            notes.append(_EXERCISE_BLOCKED_NOTE.format(reason=blocked))
    if any(f in unverifiable for f in unvalidated):
        notes.append(_UNVERIFIABLE_NOTE)
    if any(r.get("verdict") for r in runs.values()):
        notes.append(_VERDICT_NOTE)
    rows.extend(notes)

    # Bold marks the rows a reader has to act on — it is what the webview tints rows by.
    if unwritten:
        rows.append("**Declared but never written:** " + ", ".join(f"`{f}`" for f in unwritten))
    if required:
        preview = "; ".join(it["text"] for it in required[:2])
        more = f" (+{len(required) - 2} more)" if len(required) > 2 else ""
        rows.append(f"Checklist: **{_plural(len(required), 'step')} unchecked** — {preview}{more}")
    if optional:
        rows.append(f"Checklist: {_plural(len(optional), 'optional step')} not done")

    if not rows:
        return None

    chips: list[str] = []
    if written:
        chips.append(_plural(len(written), "file"))
        if unvalidated:
            chips.append(f"{len(unvalidated)} not checked")
        else:
            chips.append(f"checked: {weakest_validation_tier(execution_context, written) or 'static'}")
    if unwritten:
        chips.append(f"{len(unwritten)} declared, never written")
    if runs:
        chips.append(_plural(len(runs), "run"))
        if open_runs:
            chips.append(f"{len(open_runs)} unjudged")
        if failed:
            chips.append(f"{len(failed)} failed")
        if blocked:
            chips.append(f"{len(blocked)} not attempted")
    if required:
        chips.append(f"{_plural(len(required), 'step')} open")
    if optional:
        chips.append(f"{_plural(len(optional), 'optional step')} left")

    if unvalidated or unwritten or required or open_runs or failed:
        status = "warn"
    elif notes or optional or blocked:
        status = "note"
    else:
        status = "ok"

    return {
        "status": status,
        "files": len(written),
        "summary": " · ".join(chips),
        "rows": rows,
    }


def render_ledger(ledger: dict) -> str:
    """A built ledger as the text block appended to an answer (marker + markdown rows)."""
    # Quotes would close the marker's attributes early; nothing generated here contains
    # one, so a plain swap is enough insurance.
    summary = ledger["summary"].replace('"', "'")
    rows = "\n".join(f"- {row}" for row in ledger["rows"])
    return (
        f'\n\n{LEDGER_MARKER} status="{ledger["status"]}"'
        f' files="{ledger["files"]}" summary="{summary}"-->\n'
        f"{LEDGER_FRAMING}\n{rows}"
    )


def split_answer_ledger(text: str) -> tuple[str, str | None]:
    """Split *text* into ``(answer prose, ledger block)``; block is None when absent.

    The ledger is always the tail of an answer, so the marker's position ends the prose.
    """
    idx = text.rfind(LEDGER_MARKER)
    if idx == -1:
        return text, None
    return text[:idx].rstrip(), text[idx:].strip()


def parse_ledger_block(block: str) -> dict:
    """``{"status", "files", "summary", "rows"}`` recovered from a rendered block.

    Front-end helper (the CLI here, ``ledgerUtils.ts`` in the webview): the marker
    carries the header fields, and every row is a markdown list item.
    """
    m = _MARKER_RE.search(block)
    attrs = dict(_ATTR_RE.findall(m.group("attrs"))) if m else {}
    rows = [
        ln.strip()[2:].strip()
        for ln in block.splitlines()
        if ln.strip().startswith("- ")
    ]
    try:
        files = int(attrs.get("files", 0))
    except ValueError:
        files = 0
    return {
        "status": attrs.get("status", "note"),
        "files": files,
        "summary": attrs.get("summary", ""),
        "rows": rows,
    }
