"""What a finished or running background job says to the model: pure functions of an event.

The text states the fact and passes the payload through; the only tool names it uses
are the descriptor's ``status_op`` and ``summary_op``.
"""

from __future__ import annotations

import json

from ....servers._shared.jobs import job_key_of


# How much of a finished job's own summary a wake carries. Enough for an exit code
# and the tail of a build log; short of pasting a whole test suite into the history.
_WAKE_SUMMARY_LIMIT = 2000

# Per field, so one long value — a command built out of absolute paths, a log tail —
# cannot crowd the rest of the record out of the budget above.
_WAKE_VALUE_LIMIT = 500

# Fields the body leaves out: the head line of the wake already names the job and says
# how it ended, and repeating that as JSON makes these messages unreadable.
_WAKE_SUMMARY_SKIP = frozenset({"job_key", "kind", "state"})


def _clip(text: str) -> str:
    """One field's value, cut in the middle so both of its ends survive.

    Which end carries the meaning depends on the field — a command says it at the
    front, a log tail at the back — and this layer does not know which it is holding.
    """
    if len(text) <= _WAKE_VALUE_LIMIT:
        return text
    head = _WAKE_VALUE_LIMIT * 2 // 3
    tail = _WAKE_VALUE_LIMIT - head
    return f"{text[:head]}… [cut: {len(text)} chars] …{text[-tail:]}"


def _render_field(key: str, value: object) -> tuple[str, bool]:
    """One recorded field as ``key: value``, or as an indented block when it has lines.

    Reports whether its value was clipped, which the summary passes on: the caller
    names the tool that reads the whole of it, and only once there is more to read.
    """
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)
    clipped = _clip(text)
    was_cut = clipped != text
    text = clipped.strip("\n")
    if "\n" in text:
        body = "\n".join(f"    {line}" for line in text.splitlines())
        return f"  {key}:\n{body}", was_cut
    return f"  {key}: {text}", was_cut


def compact_summary(payload: dict) -> tuple[str, bool]:
    """A job's recorded result, one field per line, cut to a budget; and whether it was cut."""
    lines: list[str] = []
    budget = _WAKE_SUMMARY_LIMIT
    dropped = 0
    clipped = False
    for key, value in payload.items():
        if key in _WAKE_SUMMARY_SKIP or value is None or value == "":
            continue
        line, was_cut = _render_field(key, value)
        if lines and len(line) + 1 > budget:
            dropped += 1
            continue
        clipped = clipped or was_cut
        budget -= len(line) + 1
        lines.append(line)
    if dropped:
        lines.append(f"  [{dropped} more field(s) not shown]")
    if lines:
        return "\n".join(lines), bool(dropped or clipped)
    # Everything it recorded was something the head line already said.
    try:
        whole = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        whole = str(payload)
    text = _clip(whole)
    return text, text != whole


def job_complete(descriptor: dict, state: str, *, session_id: str | None = None,
                 run_session: str = "", summary: dict | None = None,
                 reason: str = "") -> dict:
    """The ``job_complete`` event of a run that reached *state*, live or found on disk.

    *session_id* is the conversation woken; *run_session* where the run's files and
    markers are (a sub-agent's run is woken in its parent), by default its origin.
    """
    status_op = descriptor.get("status_op") or {}
    return {
        "type": "job_complete",
        "job_key": job_key_of(descriptor),
        "server": descriptor.get("server"),
        "kind": descriptor.get("kind"),
        "state": state,
        "summary": summary or {},
        "status_op": status_op,
        "summary_op": descriptor.get("summary_op") or {},
        "session_id": session_id,
        "run_session": run_session or (descriptor.get("origin") or {}).get("session")
                       or session_id,
        "reason": reason if state == "unknown" else "",
    }


def wake_text(ev: dict) -> str:
    """The resume instruction of a ``job_complete`` event, from the job's own recorded result.

    Names ``summary_op`` only when the summary was cut, ``status_op`` only without one.
    """
    job_key = ev.get("job_key", "?")
    state   = ev.get("state", "done")
    kind    = ev.get("kind")
    summary = found if isinstance(found := ev.get("summary"), dict) else {}

    what = f"Background job '{job_key}'" + (f" ({kind})" if kind else "")
    if state == "crashed":
        head = f"{what} crashed."
    elif state == "unknown":
        why = ev.get("reason") or "its status stopped being readable"
        head = f"{what} can no longer be tracked: {why}."
    else:
        head = f"{what} finished."

    # Payload conventions, shown only where the server put them — keys, not tool
    # names, so a job that carries none is described by its summary alone.
    marks = []
    if summary.get("verdict"):
        marks.append(f"verdict={summary['verdict']}")
    best = summary.get("best") or {}
    if isinstance(best, dict) and best.get("primary_value") is not None:
        marks.append(f"best {summary.get('primary_metric', 'primary')}="
                     f"{best.get('primary_value')}")
    if marks:
        head = f"{head} {' '.join(marks)}"

    # A next step the *server* wrote is an instruction from something that knows
    # the job; relayed verbatim.
    next_step = summary.get("next_step")
    if next_step:
        return f"{head} {next_step}"
    if summary:
        body, cut = compact_summary(summary)
        more = ""
        if cut:
            # Only once something is actually missing, and only with a tool name the
            # descriptor handed over — the same registry data the watcher polls on.
            op = (ev.get("summary_op") or {}).get("tool")
            more = (f" What follows is cut short; read the whole of it with "
                    f"'{op}'." if op else "")
        return (f"{head} Here is what it recorded — read it, then carry on with "
                f"the work it was part of.{more}\n{body}")
    status_tool = (ev.get("status_op") or {}).get("tool")
    if status_tool:
        return (f"{head} It recorded no result of its own; read its state with "
                f"'{status_tool}', then carry on with the work it was part of.")
    return f"{head} Carry on with the work it was part of."


def checkin_text(ev: dict) -> str:
    """The check-in instruction of a ``job_checkin`` event: report briefly, investigate only if off."""
    lines = []
    for job in (ev.get("jobs") or []):
        if not isinstance(job, dict):
            continue
        key = job.get("job_key", "?")
        kind = f" ({job['kind']})" if job.get("kind") else ""
        bits = [job.get("state") or "running"]
        if job.get("phase"):
            bits.append(f"phase={job['phase']}")
        if isinstance(job.get("percent"), (int, float)):
            bits.append(f"{job['percent']}%")
        lines.append(f"- '{key}'{kind}: {', '.join(bits)}")
        alerts = job.get("alerts")
        if isinstance(alerts, dict):
            for name, line in alerts.items():
                lines.append(f"  ALERT ({name}): {line}")
        tail = job.get("log_tail")
        if isinstance(tail, list) and tail:
            lines.append("  last log lines:")
            lines.extend(f"    {ln}" for ln in tail)
    body = "\n".join(lines) or "- (no status recorded yet)"
    return (
        "Background check-in. Still running in this conversation:\n"
        f"{body}\n"
        "If this looks healthy, say so in ONE short line and stop — no tool call, "
        "no restating the plan, and do not re-answer the question these runs were "
        "launched for. If it does not — an ALERT, log lines showing the run going "
        "wrong, a status that stopped being readable, a phase that should have moved "
        "on by now — say what is wrong and what you are doing about it."
    )
