"""The completion report: what is still open when the loop ends, and its marker."""
from __future__ import annotations

import re
from ...context.signals import SOURCE_FILE_EXTENSIONS
from ..proxy_state import active_proxy_source
from ...context.execution_context import (
    failed_runs,
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
# Computed once, where the loop exits, instead of being inferred downstream from
# budgets. A retry budget with room left means "the loop would try again" only while
# the loop is running; read from the completion report it became a promise nobody was
# ever going to keep ("Checks failing (will retry)" on the last line of a finished
# run). The report below therefore speaks only in the past tense, and this says why it
# stopped speaking at all.
TERMINATION_ANSWERED: str = "answered"        # the model produced a final answer


TERMINATION_STEP_LIMIT: str = "step_limit"   # the step budget ran out


_TERMINATION_ISSUE: dict[str, str] = {
    TERMINATION_STEP_LIMIT: "Stopped: the step budget ran out before the work was finished.",
}


def _run_label(run: dict) -> str:
    """"Build" or "Run" — what hit the wall, in the user's vocabulary."""
    return "Build" if run.get("effect") == "build" else "Run"


def collect_completion_issues(
    execution_context: ExecutionContext, termination: str = TERMINATION_ANSWERED,
) -> tuple[list[str], list[str]]:
    """Return (issues, completed) describing the final completion state."""
    issues: list[str] = []
    completed: list[str] = []

    termination_issue = _TERMINATION_ISSUE.get(termination)
    if termination_issue:
        issues.append(termination_issue)

    pending = pending_validation_paths(execution_context)
    fail_counts: dict = execution_context.validation_fail_count_by_file
    dirty: set = execution_context.dirty_written_files

    if pending:
        # Split pending files into three sub-buckets for clearer feedback.
        stuck_paths = [p for p in pending if fail_counts.get(p, 0) >= VALIDATION_RETRY_BUDGET]
        retry_paths = [p for p in pending if fail_counts.get(p, 0) > 0 and fail_counts.get(p, 0) < VALIDATION_RETRY_BUDGET]
        fresh_paths = [p for p in pending if fail_counts.get(p, 0) == 0]
        if stuck_paths:
            issues.append("Checks failing, budget exhausted: " + ", ".join(stuck_paths[:5]))
        if retry_paths:
            issues.append("Checks failing, unresolved: " + ", ".join(retry_paths[:5]))
        if fresh_paths:
            issues.append("Modified files never checked: " + ", ".join(fresh_paths[:5]))
    else:
        # Name what a check actually proves. "All modified files validated" is what a
        # model reads back as licence to report the work as verified, when all it means
        # is that the files parse and are whole — the answer's correctness lives on the runs.
        tier = weakest_validation_tier(execution_context, dirty) if dirty else None
        if tier:
            completed.append(f"All modified files checked ({tier}) — parses, says nothing about the result")
        else:
            completed.append("All modified files checked")

    # A refusal is only a *blocker* when it ended the run — the hand-back case. A step
    # the user judged unnecessary is not a failure to fix, and listing it as one is what
    # made every refusal come back as "Task is incomplete". It is still reported, in its
    # own section (see refused_action_lines), because a skipped step must never be silent.
    denied_calls = execution_context.denied_tool_calls
    if denied_calls and handback_required(execution_context):
        issues.append("Stopped at the user's request: " + ", ".join(refused_action_lines(execution_context)))
    elif not denied_calls:
        completed.append("No denied actions")

    # `workflow_state` is deliberately NOT reported: it is the loop's own steering
    # variable, and "still in 'edit' state" names no gap the user can act on. Every
    # real one — a file unchecked, a run failing or unjudged, a step never started —
    # has its own line here.

    # There is deliberately no plan-vs-implementation issue here. Reporting the files
    # the checklist named and no write honoured looks like the same kind of fact as the
    # lines around it, and is not: those read a checker, a run or the disk, while this
    # one read prose the model wrote earlier and may since have thought better of. A plan
    # revised mid-course — two files dropped because the work turned out not to need
    # them — is indistinguishable at completion time from two files skipped, and the
    # declared set has no way to retract. It charged the honest revision as a gap.

    # The reminder budget stops asking after two tries; what it must not do is let a run
    # disappear. Every run still open at completion is named here, with what was tried.
    runs = execution_context.runs or {}
    never_judged = [c for c, r in sorted(runs.items()) if r.get("completed") and not r.get("verdict")]
    unresolved = [c for c, r in sorted(runs.items()) if r.get("verdict") == "unknown"]
    # A blocked run is red, and stays reported as red — but by a wall this environment put
    # there, not by the change. Charging it here is what turned "I tried the recommended
    # build and gcc is not installed" into "Task is incomplete."; see blocked_run_lines.
    broken = [(c, r) for c, r in sorted(runs.items())
              if (not r.get("completed") or r.get("verdict") == "fail")
              and not r.get("blocked")]
    # never_judged / unresolved are reported by unjudged_run_lines, not charged here: a
    # verdict is a recommendation, and charging its absence made the model spend turns
    # producing a label to clear the headline rather than because it had read something.
    # A run that actually *failed* stays an issue below — that is a machine fact.
    for command, run in broken[:5]:
        spent = int(run.get("failures", 0)) >= VALIDATION_RETRY_BUDGET
        label = "budget exhausted" if spent else "unresolved"
        issues.append(f"{_run_label(run)} failing, {label}: {run.get('command') or command}")
        # Naming the run says a wall was hit; naming the attempts says what the wall was,
        # which is the part the user needs to take it from here.
        attempts = run.get("attempts") or []
        if spent and attempts:
            issues.append("  tried: " + "; ".join(attempts[:3]))
    if runs and not never_judged and not unresolved and not broken:
        completed.append(_plural_runs(len(runs)) + " judged pass by the model")

    unchecked = [it for it in unchecked_checklist_items(execution_context) if not it.get("optional")]
    if unchecked and turn_made_commitments(execution_context):
        issues.append(
            f"Checklist incomplete: {len(unchecked)} step(s) unchecked — "
            + "; ".join(it["text"] for it in unchecked[:3])
        )

    if (
        not issues
        and not refused_action_lines(execution_context)
        and not blocked_run_lines(execution_context)
        and not unjudged_run_lines(execution_context)
    ):
        issues.append("Unknown blocker; explicit completion criteria were not met")

    return issues, completed


def unjudged_run_lines(execution_context: ExecutionContext) -> list[str]:
    """One line per run whose output was never read out, for the report.

    Same standing as :func:`blocked_run_lines`: reported, never counted. A verdict says
    what a run showed, and asking for one is a recommendation — so a missing or `unknown`
    verdict is a gap in the record, not a defect in the change. Charging it here is what
    taught the model to emit a label for every command it issued.
    """
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

    In an optimisation session the objective *is* a number, so a check that proves the
    file parses establishes nothing about the change — and `ruff` on the proxy source is
    otherwise enough to clear the check axis. This says so.

    Reported, never counted, for the reason POLICY gives for every run: a cluster, a
    queue or a dataset may simply not be there, and a gate that cannot be satisfied is
    an impasse rather than a guarantee. What an unmeasured change owes is a statement.
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


# Marker contract with the front-ends, deliberately the same shape as the verification
# ledger's (see query_engine/verification.py): an HTML comment carrying the header
# fields, then a markdown body. The front-end lifts the block off the answer and shows
# it as a collapsed disclosure — headline and risk visible, the detail one click away.
#
# The marker is what makes that possible: concatenated ahead of the answer as bare prose,
# the report has nothing to lift it by, so every front-end renders the whole of it as body
# text with the model's own words underneath — which is what makes an ordinary interim turn
# read as a wall of machine output. Nothing is dropped here, only folded.
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
    """True when a finalized answer reports the task as unfinished.

    The consumers (the CLI's re-plan offer, the sub-agent's ``completed`` flag) matched
    the headline as a literal prefix, which only worked while the report was the head of
    the answer. It is now a marked block at the tail, so they read the status the marker
    states — a run that only skipped what the user refused *is* finished, and must not be
    reported as a failure.
    """
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
    # A refusal the run stopped on stays high risk — work was cut short. A refusal the
    # run absorbed and continued past is a known, named gap: medium, never "low".
    #
    # A run left red is one too, so the level agrees with the issue the ledger prints.
    # `failed_runs` excludes blocked runs: a missing toolchain is a limitation, not a
    # defect of the change.
    # Runs merely unjudged are deliberately NOT charged here: a verdict is a
    # recommendation, and charging its absence is the trade POLICY already refused.
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
