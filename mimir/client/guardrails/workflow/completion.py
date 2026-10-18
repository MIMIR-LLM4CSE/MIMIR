"""The completion report: what is still open when the loop ends, and its marker."""
from __future__ import annotations

import re
from ...context.signals import SOURCE_FILE_EXTENSIONS
from ..proxy_state import active_proxy_source
from ...context.execution_context import (
    failed_runs,
    unsettled_runs,
    validation_tier,
    weakest_validation_tier,
)
from ...config.limits import VALIDATION_RETRY_BUDGET
from ...context.execution_context import ExecutionContext
from .commitments import turn_made_commitments, unchecked_checklist_items
from .denials import handback_required
from .state import pending_validation_paths


def _plural_runs(n: int) -> str:
    return "1 run" if n == 1 else f"{n} runs"


# ── How the loop ended ────────────────────────────────────────────────────────
# Recorded where the loop exits, so the report speaks in the past tense and says why it
# stopped, instead of promising retries that will not happen.
TERMINATION_ANSWERED: str = "answered"        # the model produced a final answer


TERMINATION_STEP_LIMIT: str = "step_limit"   # the step budget ran out


_TERMINATION_ISSUE: dict[str, str] = {
    TERMINATION_STEP_LIMIT: "Stopped: the step budget ran out before the work was finished.",
}


def _run_label(run: dict) -> str:
    """"Build" or "Run" — what hit the wall, in the user's vocabulary."""
    return "Build" if run.get("effect") == "build" else "Run"


def _validation_findings(execution_context: ExecutionContext) -> tuple[list[str], list[str]]:
    """Pending files by how far their checks got, or what the checks proved."""
    pending = pending_validation_paths(execution_context)
    if not pending:
        # Name what a check proves: "validated" reads back as licence to call the work
        # verified, when it only means the files parse and are whole.
        dirty = execution_context.dirty_written_files
        tier = weakest_validation_tier(execution_context, dirty) if dirty else None
        if tier:
            return [], [f"All modified files checked ({tier}) — parses, says nothing about the result"]
        return [], ["All modified files checked"]
    fails = execution_context.validation_fail_count_by_file
    issues = []
    for label, keep in (
        ("Checks failing, budget exhausted: ", lambda n: n >= VALIDATION_RETRY_BUDGET),
        ("Checks failing, unresolved: ", lambda n: 0 < n < VALIDATION_RETRY_BUDGET),
        ("Modified files never checked: ", lambda n: n == 0),
    ):
        paths = [p for p in pending if keep(fails.get(p, 0))]
        if paths:
            issues.append(label + ", ".join(paths[:5]))
    return issues, []


def _refusal_findings(execution_context: ExecutionContext) -> tuple[list[str], list[str]]:
    """A refusal is a blocker only when it ended the run (hand-back); otherwise it is
    reported in its own section (:func:`refused_action_lines`), never as a failure."""
    if not execution_context.denied_tool_calls:
        return [], ["No denied actions"]
    if handback_required(execution_context):
        return ["Stopped at the user's request: "
                + ", ".join(refused_action_lines(execution_context))], []
    return [], []


def _run_findings(execution_context: ExecutionContext) -> tuple[list[str], list[str]]:
    """Every failed run, with what was tried once its budget is spent.

    Unjudged and blocked runs are reported elsewhere and never charged: a verdict is a
    recommendation, and a missing prerequisite is not a defect of the change.
    """
    issues = []
    for command, run in sorted(failed_runs(execution_context).items())[:5]:
        spent = int(run.get("failures", 0)) >= VALIDATION_RETRY_BUDGET
        label = "budget exhausted" if spent else "unresolved"
        issues.append(f"{_run_label(run)} failing, {label}: {run.get('command') or command}")
        attempts = run.get("attempts") or []
        if spent and attempts:
            issues.append("  tried: " + "; ".join(attempts[:3]))
    runs = execution_context.runs or {}
    if runs and not issues and not unsettled_runs(execution_context):
        return [], [_plural_runs(len(runs)) + " judged pass by the model"]
    return issues, []


def _checklist_findings(execution_context: ExecutionContext) -> list[str]:
    unchecked = [it for it in unchecked_checklist_items(execution_context)
                 if not it.get("optional")]
    if not unchecked or not turn_made_commitments(execution_context):
        return []
    return [f"Checklist incomplete: {len(unchecked)} step(s) unchecked — "
            + "; ".join(it["text"] for it in unchecked[:3])]


def collect_completion_issues(
    execution_context: ExecutionContext, termination: str = TERMINATION_ANSWERED,
) -> tuple[list[str], list[str]]:
    """Return (issues, completed) describing the final completion state.

    ``workflow_state`` is not reported: it steers the loop and names no gap the user can
    act on; every real gap has its own line.
    """
    issues: list[str] = []
    completed: list[str] = []
    termination_issue = _TERMINATION_ISSUE.get(termination)
    if termination_issue:
        issues.append(termination_issue)
    for found, done in (_validation_findings(execution_context),
                        _refusal_findings(execution_context),
                        _run_findings(execution_context)):
        issues += found
        completed += done
    issues += _checklist_findings(execution_context)

    if (
        not issues
        and not refused_action_lines(execution_context)
        and not blocked_run_lines(execution_context)
        and not unjudged_run_lines(execution_context)
    ):
        issues.append("Unknown blocker; explicit completion criteria were not met")
    return issues, completed


def unjudged_run_lines(execution_context: ExecutionContext) -> list[str]:
    """One line per run whose output was never read out: reported, never counted."""
    runs = execution_context.runs or {}
    lines = [
        f"{run.get('command') or command} — ran; its output was never judged"
        for command, run in sorted(runs.items())
        if run.get("completed") and not run.get("verdict")
    ]
    lines += [
        f"{run.get('command') or command} — judged inconclusive"
        for command, run in sorted(runs.items()) if run.get("verdict") == "unknown"
    ]
    return lines[:5]


def unmeasured_proxy_source_lines(execution_context: ExecutionContext) -> list[str]:
    """One line when an optimisation session's source was edited and never measured.

    Reported, never counted: a cluster or a dataset may not be there.
    """

    source = active_proxy_source()
    if not source or source not in execution_context.dirty_written_files:
        return []
    if validation_tier(execution_context, source) == "measured":
        return []
    return [f"{source} — edited under an optimisation session, never measured by a run"]


def blocked_run_lines(execution_context: ExecutionContext) -> list[str]:
    """One line per run that hit a wall this environment put there, for the report.

    Reported, never counted: a prerequisite that is missing is a limitation the user has to
    see, and never a defect in the change to be charged against completion. Kept out of
    ``collect_completion_issues`` for that reason alone — it is what decides the headline.
    """
    runs = execution_context.runs or {}
    return [
        f"{_run_label(run)} not attempted — {run['blocked']}: {run.get('command') or command}"
        for command, run in sorted(runs.items()) if run.get("blocked")
    ][:5]


def refused_action_lines(execution_context: ExecutionContext) -> list[str]:
    """One line per action the user refused, for the completion report.

    Deduplicated, and named by target where there is one: three refusals of the same
    command family printed as the bare tool name three times told the user nothing
    about what they had actually refused.
    """
    lines: list[str] = []
    for item in execution_context.denied_tool_calls:
        tool_name = item.get("tool", "unknown")
        # The scope token is the command family / host / package set the refusal was
        # really about; the tool name alone is the least informative part of it.
        target = item.get("path") or (item.get("scope", "").split(":", 2) + ["", "", ""])[2]
        line = f"{tool_name}({target})" if target else tool_name
        if line not in lines:
            lines.append(line)
    return lines[:5]


# Report headlines. The wording is load-bearing: it is the one line of the report the
# user is meant to read without unfolding anything, so it lives here rather than being
# spelled out at each consumer.
HEADLINE_INCOMPLETE: str = "Task is incomplete."


HEADLINE_HANDBACK: str = "Stopped at your request."


HEADLINE_REFUSED_ONLY: str = "Task complete, except for what you refused."


# The status token the marker carries, which is what `is_incomplete_answer` reads. Kept
# beside the headlines it labels: the two are set together and must not drift.
_STATUS_BY_HEADLINE: dict[str, str] = {
    HEADLINE_INCOMPLETE:   "incomplete",
    HEADLINE_HANDBACK:     "handback",
    HEADLINE_REFUSED_ONLY: "refused-only",
}


_INCOMPLETE_STATUSES: frozenset[str] = frozenset({"incomplete", "handback"})


# Marker contract with the front-ends, shaped like the verification ledger's
# (query_engine/verification.py): an HTML comment with the header fields, then markdown.
# The front-end folds the block into a collapsed disclosure.
COMPLETION_MARKER: str = "<!--mimir:completion"


COMPLETION_FRAMING: str = "Completion report — machine-recorded, not model-authored:"


_COMPLETION_MARKER_RE = re.compile(
    re.escape(COMPLETION_MARKER) + r"(?P<attrs>[^>]*)-->")


_COMPLETION_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def render_completion_report(headline: str, risk: str, body: str) -> str:
    """The report as the block appended to an answer (marker + markdown body)."""
    # Quotes would close the marker's attributes early; nothing generated here contains
    # one, so a plain swap is enough insurance.
    safe = headline.replace('"', "'")
    return (
        f'\n\n{COMPLETION_MARKER} status="{_STATUS_BY_HEADLINE[headline]}"'
        f' risk="{risk}" headline="{safe}"-->\n'
        f"{COMPLETION_FRAMING}\n{body}"
    )


def split_answer_completion(text: str) -> tuple[str, str | None]:
    """Split *text* into ``(answer prose, completion block)``; block is None when absent.

    The counterpart of ``split_answer_ledger``. Both blocks ride at the tail, the ledger
    appended last, so a caller that has already lifted the ledger off passes the rest
    here and gets the prose.
    """
    idx = text.rfind(COMPLETION_MARKER)
    if idx == -1:
        return text, None
    return text[:idx].rstrip(), text[idx:].strip()


def parse_completion_block(block: str) -> dict:
    """``{"status", "risk", "headline", "body"}`` recovered from a rendered block.

    Front-end helper, mirroring ``parse_ledger_block``: the marker carries the header
    fields and everything past the framing line is the markdown body.
    """
    m = _COMPLETION_MARKER_RE.search(block)
    attrs = dict(_COMPLETION_ATTR_RE.findall(m.group("attrs"))) if m else {}
    body = block[m.end():] if m else block
    body = body.lstrip("\n")
    if body.startswith(COMPLETION_FRAMING):
        body = body[len(COMPLETION_FRAMING):]
    return {
        "status":   attrs.get("status", "incomplete"),
        "risk":     attrs.get("risk", ""),
        "headline": attrs.get("headline", ""),
        "body":     body.strip(),
    }


def is_incomplete_answer(answer: str) -> bool:
    """True when a finalized answer's status marker reports the task unfinished."""
    _, block = split_answer_completion(answer)
    if block is None:
        return False
    return parse_completion_block(block)["status"] in _INCOMPLETE_STATUSES


def finalize_incomplete_answer(
    answer: str, execution_context: ExecutionContext, termination: str = TERMINATION_ANSWERED,
) -> str:
    issues, completed = collect_completion_issues(execution_context, termination)
    pending = pending_validation_paths(execution_context)
    denied_calls = execution_context.denied_tool_calls
    refused = refused_action_lines(execution_context)
    handback = handback_required(execution_context)

    # Refusals alone do not make a task incomplete: reading (2) of a refusal is "this
    # step was not needed", and the honest report of that is a completed task with a
    # named omission — not a failure. Only a hand-back, some *other* open issue, or a
    # run that never reached a final answer ends the run unfinished.
    if handback:
        headline = HEADLINE_HANDBACK
    elif refused and not issues and termination == TERMINATION_ANSWERED:
        headline = HEADLINE_REFUSED_ONLY
    else:
        headline = HEADLINE_INCOMPLETE

    sections: list[str] = ["Completed:\n- " + "\n- ".join(completed)]
    blocked = blocked_run_lines(execution_context)
    if blocked:
        sections.append(
            "Not attempted (a prerequisite this environment does not have):\n- "
            + "\n- ".join(blocked)
        )
    # Deliberately NOT listed here: the ledger appended to this same answer already
    # carries one row per unjudged run, in the same words. Said twice — capped at five
    # above and uncapped below — it was the largest single thing in the report, and the
    # second copy told the reader nothing the first had not. `unjudged_run_lines` is
    # still what decides, below, whether there is anything to report at all.
    unmeasured = unmeasured_proxy_source_lines(execution_context)
    if unmeasured:
        sections.append("Checked but never measured:\n- " + "\n- ".join(unmeasured))
    if issues:
        sections.append("Remaining issues:\n- " + "\n- ".join(issues))
    if refused and not handback:
        sections.append(
            "Not performed (you refused these; they were skipped, not attempted "
            "another way):\n- " + "\n- ".join(refused)
        )

    # Only treat unvalidated source-code files as high-risk; unvalidated non-code
    # files (e.g. .md, .txt) that never go through a code validator are low-risk.
    _pending_code = [p for p in pending if any(p.endswith(ext) for ext in SOURCE_FILE_EXTENSIONS)]
    # A refusal the run stopped on is high risk; one it continued past, and a run left red
    # (blocked runs excluded), are medium. Unjudged runs are not charged.
    _failed_runs = failed_runs(execution_context)
    # An optimisation source that was checked but never measured is a named gap, not a
    # defect: medium, never "low".
    risk_level = (
        "high" if _pending_code or handback
        else ("medium" if pending or denied_calls or _failed_runs or unmeasured
              else "low")
    )
    sections.append(f"Residual risk: {risk_level}.")

    # The model's own prose is the answer, and the machine's account of the same run rides
    # behind the marker. This order and not the reverse: the report first and the prose last
    # is what a reader meets head-on on an ordinary interim turn. The framing is the
    # marker's own ("machine-recorded, not model-authored"), stated once, where the
    # machine's lines actually are.
    return answer.strip() + render_completion_report(
        headline, risk_level, "\n\n".join(sections))
