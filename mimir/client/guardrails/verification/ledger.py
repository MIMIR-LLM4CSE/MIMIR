"""The verification ledger: what a run established, machine-recorded and appended to the answer.

It opens with a marker line (status, summary) that front-ends fold
(:func:`split_answer_ledger`). Report-only: built after the model stopped acting.
"""
from __future__ import annotations

import re

from collections import Counter

from ...context import RUN_CLASSES, run_class, validation_tier, weakest_validation_tier
from ..workflow import unchecked_checklist_items
from ...context.execution_context import ExecutionContext

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
    "A check says a file parses and is whole. It says nothing about whether the "
    "answer is right, and nothing here was built or run — so no result was produced, and "
    "none was judged."
)
# Not a gap the model can close, and now the only one of its kind: the mandatory check
# runs in-process, so a file escapes it only by not being text at all.
_UNVERIFIABLE_NOTE = (
    "These files are not readable as text here, so the check that is otherwise "
    "required could not be run on them."
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


#: How much of a command a row shows. A row's job is to *identify* the run, not to
#: reproduce it — the transcript keeps the text in full. Left unbounded, a single row
#: reached 896 characters of shell (a multi-line ``python -c`` probe), and a ledger of
#: those is a paste rather than a record.
_COMMAND_WIDTH = 110

#: How many run rows a ledger names before it stops and gives a count. The chips
#: already carry the totals, so past this the enumeration only repeats them at length:
#: one turn listed 22 runs the summary had already counted.
_MAX_RUN_ROWS = 6



def _short_command(command: str) -> str:
    """*command* on one line, cut to :data:`_COMMAND_WIDTH`."""
    flat = " ".join(str(command).split())
    return flat if len(flat) <= _COMMAND_WIDTH else flat[:_COMMAND_WIDTH - 1] + "…"


def _run_row(command: str, run: dict) -> str:
    """One execution, as what is actually known about it."""
    command = _short_command(command)
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


def _file_rows(execution_context: ExecutionContext, written: list[str],
               unvalidated: list[str]) -> list[str]:
    unverifiable = execution_context.checks.unverifiable_files or set()
    rows = []
    for f in written:
        if f not in unvalidated:
            tier = validation_tier(execution_context, f) or "structural"
            rows.append(f"`{f}` — checked: {tier}")
        elif f in unverifiable:
            rows.append(f"`{f}` — **not checked** (not readable as text)")
        else:
            rows.append(f"`{f}` — **not checked**")
    return rows


def _run_rows(runs: dict) -> list[str]:
    """Runs by what the reader has to act on, cut at :data:`_MAX_RUN_ROWS`.

    The row shows the command as typed, not its ledger key; the chips keep the totals.
    """
    ordered = sorted(runs.items(),
                     key=lambda kv: (RUN_CLASSES.index(run_class(kv[1])), kv[0]))
    rows = [_run_row(run.get("command") or command, run)
            for command, run in ordered[:_MAX_RUN_ROWS]]
    if len(ordered) > _MAX_RUN_ROWS:
        hidden = len(ordered) - _MAX_RUN_ROWS
        rows.append(f"{_plural(hidden, 'further run')} not listed — counted above")
    return rows


def _notes(execution_context: ExecutionContext, written: list[str],
           unvalidated: list[str], runs: dict) -> list[str]:
    notes = []
    if written and not unvalidated and not runs:
        notes.append(_UNCHECKED_OUTPUT_NOTE)
        reason = str(execution_context.execution.exercise_blocked_reason or "").strip()
        if reason:
            notes.append(_EXERCISE_BLOCKED_NOTE.format(reason=reason))
    if set(unvalidated) & (execution_context.checks.unverifiable_files or set()):
        notes.append(_UNVERIFIABLE_NOTE)
    if any(r.get("verdict") for r in runs.values()):
        notes.append(_VERDICT_NOTE)
    return notes


def _checklist_rows(required: list[dict], optional: list[dict]) -> list[str]:
    # Bold marks the rows a reader has to act on: the webview tints rows by it.
    rows = []
    if required:
        preview = "; ".join(it["text"] for it in required[:2])
        more = f" (+{len(required) - 2} more)" if len(required) > 2 else ""
        rows.append(f"Checklist: **{_plural(len(required), 'step')} unchecked** — {preview}{more}")
    if optional:
        rows.append(f"Checklist: {_plural(len(optional), 'optional step')} not done")
    return rows


def _chips(execution_context: ExecutionContext, written: list[str], unvalidated: list[str],
           runs: dict, by_class: Counter, required: list[dict],
           optional: list[dict]) -> list[str]:
    chips: list[str] = []
    if written:
        chips.append(_plural(len(written), "file"))
        if unvalidated:
            chips.append(f"{len(unvalidated)} not checked")
        else:
            tier = weakest_validation_tier(execution_context, written) or "static"
            chips.append(f"checked: {tier}")
    if runs:
        chips.append(_plural(len(runs), "run"))
        open_runs = by_class["unknown"] + by_class["unjudged"]
        for count, label in ((open_runs, "unjudged"), (by_class["broken"], "failed"),
                             (by_class["blocked"], "not attempted")):
            if count:
                chips.append(f"{count} {label}")
    if required:
        chips.append(f"{_plural(len(required), 'step')} open")
    if optional:
        chips.append(f"{_plural(len(optional), 'optional step')} left")
    return chips


def build_ledger(execution_context: ExecutionContext) -> dict | None:
    """``{"status", "files", "summary", "rows"}`` for this run, or None if nothing happened.

    ``status`` is ok, note or warn. File rows say what a checker established; run rows what
    an execution showed and what the model read.
    """
    written = sorted(execution_context.edits.dirty_written_files)
    unchecked = unchecked_checklist_items(execution_context)
    runs = execution_context.execution.runs or {}
    # A run nobody judged is worth reporting even when no file was touched.
    if not written and not unchecked and not runs:
        return None

    required = [it for it in unchecked if not it.get("optional")]
    optional = [it for it in unchecked if it.get("optional")]
    unvalidated = [f for f in written if f not in execution_context.checks.validated_files]
    by_class = Counter(run_class(r) for r in runs.values())
    notes = _notes(execution_context, written, unvalidated, runs)
    rows = (_file_rows(execution_context, written, unvalidated) + _run_rows(runs)
            + notes + _checklist_rows(required, optional))
    if not rows:
        return None

    if unvalidated or required or by_class["unknown"] or by_class["unjudged"] \
            or by_class["broken"]:
        status = "warn"
    elif notes or optional or by_class["blocked"]:
        status = "note"
    else:
        status = "ok"
    chips = _chips(execution_context, written, unvalidated, runs, by_class,
                   required, optional)
    return {"status": status, "files": len(written), "summary": " · ".join(chips),
            "rows": rows}


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
