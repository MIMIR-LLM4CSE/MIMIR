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

from ..context import unwritten_declared_files, validation_tier, weakest_validation_tier
from ..guardrails.workflow import unchecked_checklist_items

#: Opening marker of a rendered ledger block. Stable — UIs split answers on it.
LEDGER_MARKER = "<!--mimir:ledger"

_MARKER_RE = re.compile(r"<!--mimir:ledger(?P<attrs>[^>]*)-->")
_ATTR_RE = re.compile(r'(?P<key>\w+)="(?P<value>[^"]*)"')

#: Kept as the block's first prose line so the framing reaches the model even when
#: a front-end drops the marker comment.
LEDGER_FRAMING = "Verification ledger — machine-recorded, not model-authored:"

# How a file reached the `oracle` tier. Derived, never stored: a file promoted by
# red→green is exactly one an `executed`-tier check was seen failing on, and that set
# is already kept for the promotion itself. Anything else at `oracle` got there by
# reporting a numerical invariant.
_ORACLE_DISCRIMINATED = "red→green"
_ORACLE_INVARIANT = "reported invariant"

# The absence line is domain-neutral on purpose: worded numerically it fires on every
# parser/CLI/refactor run that could never satisfy it, and becomes wallpaper.
_DISCRIMINATION_NOTE = (
    "Discrimination: none observed — every check that passed was first run after the "
    "change and was never seen failing, so nothing establishes that it tells working "
    "code from broken code."
)
# Numerical wording only ever *qualifies* an invariant that was actually reported.
_INVARIANT_NOTE = (
    "Reported invariant: a validation run printed a numerical invariant. Its presence "
    "is recorded, never its value — nothing here compared it against a sealed reference."
)


def _oracle_basis(execution_context: dict, path: str) -> str:
    if path in (execution_context.get("executed_failures") or set()):
        return _ORACLE_DISCRIMINATED
    return _ORACLE_INVARIANT


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def build_ledger(execution_context: dict) -> dict | None:
    """The ledger for this run, or None when nothing happened worth recording.

    Returns ``{"status": ok|note|warn, "files": int, "summary": str, "rows": [markdown]}``.
    ``status`` separates a clean run from soft caveats (evidence that passed but
    discriminates nothing) and from hard gaps (unvalidated or unwritten files, open
    checklist steps) — it is what a front-end colours the panel by.
    """
    written = sorted(execution_context.get("dirty_written_files", set()))
    validated = execution_context.get("validated_files", set())
    unchecked = unchecked_checklist_items(execution_context)
    unwritten = unwritten_declared_files(execution_context)
    required = [it for it in unchecked if not it.get("optional")]
    optional = [it for it in unchecked if it.get("optional")]

    if not written and not unchecked and not unwritten:
        return None

    rows: list[str] = []
    notes: list[str] = []
    unvalidated = [f for f in written if f not in validated]

    for f in written:
        if f in unvalidated:
            rows.append(f"`{f}` — **not validated**")
            continue
        tier = validation_tier(execution_context, f) or "executed"
        if tier == "oracle":
            # Saying plainly which way it got there is what stops "the tests passed"
            # being read back as "the result is correct".
            rows.append(f"`{f}` — validated: oracle ({_oracle_basis(execution_context, f)})")
        else:
            rows.append(f"`{f}` — validated: {tier}")

    tiers = [validation_tier(execution_context, f) for f in written]
    if written:
        # Only worth saying once code was run — below that the per-file tier tells the
        # whole story.
        if "oracle" not in tiers and "executed" in tiers:
            notes.append(_DISCRIMINATION_NOTE)
        elif any(_oracle_basis(execution_context, f) == _ORACLE_INVARIANT
                 for f in written if validation_tier(execution_context, f) == "oracle"):
            notes.append(_INVARIANT_NOTE)
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
            chips.append(f"{len(unvalidated)} not validated")
        else:
            weakest = weakest_validation_tier(execution_context, written)
            chips.append(f"evidence: {weakest or 'executed'}")
    if unwritten:
        chips.append(f"{len(unwritten)} declared, never written")
    if required:
        chips.append(f"{_plural(len(required), 'step')} open")
    if optional:
        chips.append(f"{_plural(len(optional), 'optional step')} left")

    if unvalidated or unwritten or required:
        status = "warn"
    elif notes or optional:
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
