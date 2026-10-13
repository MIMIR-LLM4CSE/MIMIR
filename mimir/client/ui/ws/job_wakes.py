"""What a finished or still-running background job says to the model.

Pure functions of one event, held apart from both consumers because there are two and
they must say the same thing. A socket's ``_Session`` turns a wake into a turn while
somebody is attached; the pool does it when nobody is
(:meth:`_AgentPool.consume_durable_event`), and a run reported through one route must be
indistinguishable from the same run reported through the other — the model has no way to
tell which consumer woke it, and a wake worded differently by each is a conversation
whose history depends on whether its user had the window open.

Nothing here interprets a job. The client does not know what a detached run *does* — it
may be a two-hour compile, a Slurm batch, a proxy optimization — so the text states the
fact and passes the recorded payload through under the server's own field names. The
only tool names any of this may utter are ``status_op``'s and ``summary_op``'s, and only
because those are registry data travelling on the descriptor.
"""

from __future__ import annotations

import json

# Events that describe a detached run rather than a turn. Everything else a worker
# queues — its tokens, its rows, its answer — is debris once the socket drawing it is
# gone; these two are the reason the pool consumes the bus at all. They have to reach a
# turn whether or not anyone is attached, so both consumers key off this one set.
DURABLE_EVENTS = frozenset({"job_complete", "job_checkin"})

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
    """A job's recorded result, as one field per line, cut to a budget.

    Passed through rather than interpreted: the client does not know what kind of job
    ran, so it hands the model what the server recorded, under the server's own field
    names. What it chooses is the shape — lines instead of a single JSON string, each
    value clipped on its own — because a wake is read by a person as well as a model,
    and one 800-character command should not be the whole of what either sees.

    Returns the text and whether it had to be cut — the caller says where the whole of
    it can be read, which is only honest once something is missing.
    """
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


def wake_text(ev: dict) -> str:
    """Build the auto-resume instruction from a ``job_complete`` event.

    The wake states the fact and passes the payload through: what the model should do
    next comes from the job's own recorded result, never from an instruction invented
    here. Naming another server's ops in this function is how a build once got told to
    review proxy results and continue an optimization loop that did not exist.

    The only tool names it may use are ``status_op``'s and ``summary_op``'s, and only
    because those are registry data travelling on the descriptor — the same reason the
    watcher can poll generically. The first is the last resort, for a job that recorded
    no summary; the second is named only when the summary had to be cut, so that what
    was left out is still reachable.
    """
    job_key = ev.get("job_key", "?")
    state   = ev.get("state", "done")
    kind    = ev.get("kind")
    summary = ev.get("summary") if isinstance(ev.get("summary"), dict) else {}

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
    """Build the check-in instruction from a ``job_checkin`` event.

    Under the same rule as :func:`wake_text` — it names no tool of any server and
    interprets nothing, passing on whatever the status op chose to report. What it
    adds is a ceiling on the answer: a check-in exists so a run that went wrong early
    is not discovered at its end, not to restate the plan or go and look at something.

    Reading the run is still open to it where the status says something is off: the
    dispatch guard refuses the *status* op of a watched job (this event already
    carries that answer) and leaves the summary op alone.
    """
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
