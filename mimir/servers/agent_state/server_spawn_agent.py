"""
MCP Spawn-Agent Server
======================
Lets the orchestrator delegate a sub-task to a fresh MimirAgent instance that
runs to completion and returns its answer as a string.  Multiple spawn_agent
calls emitted in a single model step run **concurrently**: the parent gathers
them, and this tool is async so a running child never blocks this server's loop
(a sync tool is awaited inline by FastMCP, which would serialize the fan-out).

While a child runs, its own tool calls are streamed back to the caller as MCP
progress notifications, so the UI can show what the child is doing instead of one
row spinning for the whole cap.

Tools:
  1. spawn_agent(task, context?, tools?, max_steps?, time_budget_secs?, model?)
        — spin up a child MimirAgent, run it, return its answer.

The caller names the tools the child gets, out of what it may itself grant (sent with
the call: its own visible tools, less the ones reserved to it). The child connects
only the servers those tools live in, keeps only those tools, and runs read-only
unless one of them writes or executes.

The child runs on the caller's model unless the caller names another one the
endpoint serves. What each served model is worth is read once, at start-up, from the
client's model catalog and becomes the description of the ``model`` argument.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Annotated, Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
# Run as a subprocess MCP server, the package root is not on the path yet. Written
# without a name for the path: a bare assignment here is code before the imports, and
# ruff reads everything after it as an import out of place (E402), where it lets the
# sys.path calls a server needs stand.
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) not in sys.path:
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from approved_roots import approved_roots
from capabilities import BACKGROUNDABLE, DELEGATE, MAIN_ONLY, REVERSIBLE, tool_caps
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from responses import err, ok

mcp = FastMCP("spawn_agent")

# The read-only mode a child without any writing tool runs in (one of the client's
# config.models.READONLY_MODES). "ask" rather than "plan": the child answers a
# question, it does not draft a plan.
_READONLY_CHILD_MODE = "ask"

# Wall for one sub-agent run, enforced here and set per call by the caller. What the
# tool DECLARES to the dispatcher is the maximum plus a margin, so this cap fires
# first and hands back what the child had instead of the parent killing the call with
# nothing to show. The maximum stays under the client's own ceiling on a declared
# timeout (config.constants.TOOL_CALL_TIMEOUT_MAX_SECS, 1200s).
SUBAGENT_DEFAULT_BUDGET_SECS = 600
SUBAGENT_MIN_BUDGET_SECS = 60
SUBAGENT_HARD_CAP_SECS = 1140

# The request-_meta key the client sends its approval mode under (client
# guardrails.policy.approval.APPROVAL_MODE_META). A child runs in that mode; absent or
# unknown, it runs in "manual" — the mode that lets nothing sensitive through unasked.
_APPROVAL_MODE_META = "mimir/approval_mode"
_APPROVAL_MODES = ("manual", "auto", "auto_all")

# The request-_meta key the client sends its current model under (client
# config.models.CALLER_MODEL_META). A child with no model named runs on it.
_CALLER_MODEL_META = "mimir/model"

# What the caller may hand over, and the mode it was computed under (client
# config.models.GRANTABLE_TOOLS_META / CALLER_MODE_META). The table is {tool: owning
# server}: the child connects servers, not tools. Absent, only an exploration with no
# named tool runs — a caller that cannot say what it may grant grants nothing.
_GRANTABLE_TOOLS_META = "mimir/grantable"
_GRANTABLE_WRITERS_META = "mimir/grantable_writers"
_CALLER_MODE_META = "mimir/mode"

# How far the user lets a sub-agent go (client config.constants.SUBAGENT_LEVELS,
# config.models.SUBAGENT_LEVEL_META). A rung hands something over, so an absent or
# unknown one is read as the weakest: a child that only reads.
_SUBAGENT_LEVELS = ("explore", "parallel")
_DEFAULT_SUBAGENT_LEVEL = _SUBAGENT_LEVELS[0]
_SUBAGENT_LEVEL_META = "mimir/subagent_level"


def _probe_served_models() -> list[dict]:
    """The models the endpoint serves, asked once at start-up; [] when it lists none.

    Once, so the ``model`` description below is the same on every turn and the
    prompt prefix stays cacheable; a model added to the endpoint mid-session is
    offered from the next session on.
    """
    try:
        from mimir.client.query_engine.backends.factory import get_backend
        return get_backend().served_model_info()
    except Exception as exc:  # a dead endpoint must not keep the server from starting
        print(f"spawn_agent: could not list served models: {exc}", file=sys.stderr)
        return []


def _model_description(served: list[dict]) -> str:
    try:
        from mimir.client.config.model_catalog import describe_served_models
        return describe_served_models(served)
    except Exception as exc:
        print(f"spawn_agent: model catalog unavailable: {exc}", file=sys.stderr)
        return "Which model the sub-agent runs on. Leave it empty to use your own model."


_SERVED_MODELS: list[dict] = _probe_served_models()
_MODEL_DESCRIPTION = _model_description(_SERVED_MODELS)

# Constant for the session, like every other tool description, so the prompt prefix
# stays cacheable: it names no tool and no count, because what may actually be granted
# depends on the caller's mode and the user's toggles at the moment of the call.
_TOOLS_DESCRIPTION = (
    "The tools the sub-agent may use, named exactly as they appear in your own tool "
    "list. Leave it empty for read-only reconnaissance: the child then gets reading, "
    "search and code navigation, and can run nothing. Name tools and it gets those and "
    "nothing else, so give it what its task needs — a writing or executing tool only "
    "when the task must write or run something, and then it works with your approval "
    "mode and keeps a todo list of its own. Some tools are yours alone and are refused "
    "here: planning, delegation, asking the user, writing memory, cluster submission. "
    "A sub-agent you give a writing or running tool works in a COPY of the repository, "
    "on its own branch, and hands back a branch and a diff — never merged for you; so "
    "several of them can work at once without touching each other's files. "
    "A name you cannot grant is refused with the list of what you can, so ask for what "
    "the task needs rather than guessing small."
)


def _human_budget(secs: int) -> str:
    minutes = secs // 60
    return f"about {minutes} minutes" if minutes >= 2 else f"{secs} seconds"


def _prune_tools(agent, granted: list[str]) -> None:
    """Keep only *granted* on this child agent, whatever its servers advertised.

    A server is a bundle: connecting the one that owns the granted tool brings its
    siblings too. Dropping them here is what makes the grant exact — the model never
    sees them, and the dispatcher answers "unknown tool" to a name that is not in
    ``tool_owner``, so a hallucinated call fails closed rather than acting.
    """
    keep = set(granted)
    agent.tools = [t for t in agent.tools if t.get("function", {}).get("name") in keep]
    agent.tool_owner = {k: v for k, v in agent.tool_owner.items() if k in keep}
    agent.tool_caps = {k: v for k, v in agent.tool_caps.items() if k in keep}


def _write_record(session: str, record: dict) -> None:
    """Leave a card for this sub-agent in its own session directory.

    One small ``subagent.json`` per child: what it was asked, what it was given, how it
    ended. The sub-agents panel reads these — a run that is over still has to be
    answerable for, and its todo list next door says nothing about who wrote it. Best
    effort: a child's work must not fail over its own bookkeeping.
    """
    if not session:
        return
    try:
        from state_paths import state_dir
        directory = os.path.join(state_dir(), "sessions", session)
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, "subagent.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(directory, "subagent.json"))
    except Exception as exc:
        print(f"spawn_agent: could not record sub-session {session}: {exc}", file=sys.stderr)


def _repo_relative(paths: list[str], worktree: dict | None) -> list[str]:
    """Paths as the *repository* names them, not as this child's copy does.

    A child given a worktree reads and writes under /tmp, and reporting that path back
    makes evidence the caller cannot quote and a link that dies with the copy. The git
    side of the same report (``files_changed``) has always been repo-relative; this is
    the rest of it agreeing. A path outside the copy — the caller's own tree, /tmp — is
    left exactly as it is.
    """
    root = (worktree or {}).get("path") or ""
    if not root:
        return paths
    root = os.path.join(os.path.realpath(root), "")
    out = []
    for path in paths:
        real = os.path.realpath(path)
        out.append(real[len(root):] if real.startswith(root) else path)
    return out


def _abandon_child(child: dict) -> None:
    """Tell a child that overran its budget to stop, and mark its record abandoned.

    The child runs ``asyncio.run()`` on a thread of its own, so there is nothing here
    to cancel — only the cooperative flag its loop already honours (``_cancel_flag``,
    read by the backend mid-stream and at every step boundary). Setting it aborts the
    turn in flight; the child's own ``finally`` then writes its record, and the mark
    left here is how that write tells a run its caller dropped from one it waited for.

    Without this the thread went on spending model calls and tool budget on an answer
    nobody would read, and landed its card as a clean "finished" minutes after the
    caller had given up on it — two sub-agents on one task, competing for one endpoint.

    Best effort by design: the flag is read from another thread, so a child already
    between steps stops one step later, and one mid-teardown may not stop at all.
    """
    child["abandoned"] = True
    flag = getattr(child.get("agent"), "_cancel_flag", None)
    if flag is None:
        return
    try:
        flag.set()
    except Exception as exc:  # a child mid-teardown must not fail its caller
        print(f"spawn_agent: could not stop abandoned sub-agent: {exc}",
              file=sys.stderr)


def _final_state(child: dict | None, result: dict | None) -> str:
    """The terminal state to record for a finished child.

    "abandoned" outranks the other two: a child whose caller timed out may still
    reach a clean result seconds later, and writing that as "finished" tells the
    panel a run succeeded when its answer went nowhere.
    """
    if child is not None and child.get("abandoned"):
        return "abandoned"
    return "finished" if result else "failed"


def _partial_handoff(child: dict) -> dict:
    """What can be read off a child still running past its budget.

    Its answer is gone — it is mid-turn in another thread, and nothing can ask it to
    conclude. What it touched is already recorded on the agent, and saying where it
    got to is the difference between a caller that can carry the axis on and one that
    must start it again.
    """
    agent = child.get("agent")
    carry = getattr(agent, "_carry_context", None) or {}
    worktree = child.get("worktree")
    files_read = _repo_relative(sorted(carry.get("read_files") or []), worktree)
    files_written = _repo_relative(
        sorted(carry.get("last_query_written_files") or []), worktree)
    lines = []
    if files_written:
        lines.append("Files it had already modified: " + ", ".join(files_written) + ".")
    if files_read:
        lines.append(f"It had read {len(files_read)} file(s).")
    lines.append("Its todo list and scratchpad are under its session, as it left them.")
    return {
        "answer": " ".join(lines),
        "files_read": files_read,
        "files_written": files_written,
    }

# Ceiling on what one step of a child may generate. The backend otherwise grants the
# model's whole answer reserve, and a run that spends it is a single silent step.
SUBAGENT_ANSWER_TOKENS = 8192

# How much of the model's window one child may fill, by what the child was given.
# Not the model's whole window: a child exists to keep its reading out of the caller's
# context, and several of them on one endpoint each budgeting for the full window is
# how a fan-out becomes the thing it was meant to avoid. Both numbers are ceilings —
# a model with a smaller window still sizes the budget, and the trimming, eviction and
# compaction all follow from the total.
#
# An explorer reads a handful of files and returns a conclusion, and a sweep wide
# enough to be sure is what it is for — twice the compact budget, so breadth is not
# what makes it hand back a half-answer. A working child is the other case — it reads
# what it must to change code, runs it, and reports what came back — so it gets the
# standing full-mode budget rather than a share of it.
SUBAGENT_CONTEXT_TOKENS_EXPLORE = 64_000
SUBAGENT_CONTEXT_TOKENS_WORKING = 200_000

# What an explorer owes back. Its own mode prompt already asks for cited prose; this
# says the part that is about the *parent*: a conclusion costs the caller a paragraph
# of context, the file contents it read would cost the window they were meant to save.
_EXPLORE_BRIEF = (
    "You are an exploration sub-agent. Answer the question below by reading the code, "
    "and return a CONCLUSION: what you found, with the concrete file paths, symbols and "
    "line numbers that back each claim. Do not paste the file contents you read — the "
    "caller wants your finding, not your reading. Say plainly what you could not "
    "establish rather than guessing. You own the breadth of the sweep: search widely "
    "enough to be sure, then stop."
)

# Said to a child working in a copy of its own. Three facts it cannot infer: where it
# is, that its starting point is the last commit rather than the caller's current files,
# and that its work is kept by the commit MIMIR makes for it — not by the directory,
# which goes when the axis ends.
_COPY_BRIEF = (
    "\n\nYou work in your OWN copy of the repository at {path}, on branch {branch}. It "
    "starts from the last commit, so uncommitted changes elsewhere are not here. Other "
    "sub-agents work on other copies at the same time: what you build, run and measure "
    "is yours alone, and nothing you do touches their files. When you finish, your "
    "changes are committed to your branch for you and this directory is removed — so "
    "leave the work in the files, and say in your answer what you changed and what it "
    "measured. If this task involves a measurement proxy, register one of your own "
    "pointing at THIS copy; the sealed reference and the benchmark suite are shared, so "
    "do not seal them again."
)

# What a working child owes back. It has its own todo list and its own scratchpad, and
# it may be one of several axes running at once — so what the caller needs from it is
# not only the result but the state: enough to carry the axis on with a fresh child
# rather than from the beginning.
_WORK_BRIEF = (
    "You are a working sub-agent: one self-contained piece of a larger task, with your "
    "own context and your own todo list. Only the tools listed for you are available — "
    "nothing else is connected, so plan within them and say so if the task truly needs "
    "more. You have {budget} and at most {steps} tool-call steps: before they run out, "
    "STOP and finish with a HANDOFF — what you established and how you verified it, what "
    "is left, and where it is (files, branch, scratchpad). A caller who gets your handoff "
    "can carry the work on; a caller who gets nothing pays for it twice."
)

# ── Sub-agent output routing ──────────────────────────────────────────────────
# This server is a subprocess whose stdout IS the JSON-RPC pipe, so anything
# printed here corrupts the protocol. Two leaks are plugged: the child agent gets
# an event sink (without one, event_sink.emit falls back to printing every event),
# and stdout is pointed at stderr for whatever still prints.

_CHILD_QUEUE_MAX = 256      # child events buffered between two forwarding ticks
_MAX_EVENTS_PER_TICK = 8    # forwarded per tick, so a tool storm cannot hog the loop
_MAX_EVENTS_PER_RUN = 500   # ceiling per child run; past it only the trailer is sent
_POLL_SECS = 0.05
# The child's activity log: enough to see what it has been doing, bounded so a long
# run cannot fill a disk with rows nobody will scroll back to.
_ACTIVITY_MAX_BYTES = 256_000
_ACTIVITY_KEEP_LINES = 400
# A child in a model turn emits nothing at all, and the caller's row has no way to
# tell that from a hung run. Past this silence, say it is still there.
_HEARTBEAT_SECS = 20.0

_stdout_silenced = False


def _silence_stdout_once() -> None:
    """Point stdout at stderr, once, on first use — never at import time.

    The stdio transport wraps ``sys.stdout.buffer`` once at startup; rebinding
    before that would make it wrap *stderr* and the server would go mute. Doing it
    afterwards is harmless (the transport holds its own reference), and it is never
    restored: two concurrent children would restore each other's saved value.
    """
    global _stdout_silenced
    if _stdout_silenced:
        return
    sys.stdout = sys.stderr
    _stdout_silenced = True


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_event(ev: dict) -> dict | None:
    """One child engine event in wire form, or None when it is not worth forwarding.

    The child's tool activity travels, and its step-level status lines with it:
    tokens, thinking and diffs would cost far more than they show, and the caller
    renders the rest as ordinary tool rows. Keys are short because each event rides
    in a progress notification.

    Status matters because a step is not one model call. An empty turn is retried, a
    nudge is answered, a checklist refresh re-asks — each up to the child's whole
    per-step ceiling, none of them a tool call. Dropped, they left a gap in the log
    with nothing in it, and a child spending four minutes that way was indistinguishable
    from a child that had hung. The loop already says what it is doing; forwarding it
    is the difference between reading that and guessing at it.
    """
    kind = ev.get("type")
    if kind == "tool_call":
        return {
            "v": 1, "t": "tc",
            "i": str(ev.get("id") or ""),
            "n": str(ev.get("name") or ""),
            "l": _clip(ev.get("label"), 120),
            "d": _clip(ev.get("detail"), 160),
        }
    if kind == "tool_result":
        out = {
            "v": 1, "t": "tr",
            "i": str(ev.get("id") or ""),
            "ok": bool(ev.get("ok")),
            "s": _clip(ev.get("summary"), 160),
        }
        ms = ev.get("duration_ms")
        if isinstance(ms, (int, float)):
            out["ms"] = int(ms)
        if isinstance(ev.get("target"), dict):
            out["f"] = ev["target"]
        return out
    if kind == "status":
        text = _clip(ev.get("text"), 160).strip()
        # The loop emits a blank status as a spacer for a terminal that has rows to
        # separate. This log has none, and a blank line here reads as a lost event.
        return {"v": 1, "t": "st", "s": text} if text else None
    return None


def _activity_path(session: str) -> str:
    from state_paths import state_dir
    return os.path.join(state_dir(), "sessions", session, "activity.jsonl")


def _append_activity(session: str, event: dict) -> None:
    """Add one line to this child's activity log, trimming it when it grows.

    The log is how anyone watches a sub-agent work. The progress channel only exists
    while the call is open — a detached child would otherwise be invisible until it
    lands — and a file is read the same way by both kinds of child, by the panel, and
    after the fact. Best effort: a child's work must not fail over its own log.
    """
    if not session:
        return
    try:
        path = _activity_path(session)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**event, "at": time.time()}, separators=(",", ":")) + "\n")
            trim = fh.tell() > _ACTIVITY_MAX_BYTES
        if trim:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()[-_ACTIVITY_KEEP_LINES:]
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            os.replace(tmp, path)
    except Exception:
        pass


def _make_child_sink(q: queue.Queue, counters: dict,
                     job_key: str = "", session: str = "") -> Callable[[dict], None]:
    """The child's event_callback: enqueue, never block, never raise.

    Raising here would put the child's engine back on emit()'s print fallback —
    straight into the JSON-RPC pipe — so a full queue drops the event and counts it.

    Three readers, one sink. The caller's progress channel gets every event while the
    call is open; the child's activity log gets them all regardless, which is what lets
    anyone watch a detached child; and a detached job records its last tool row as its
    phase, the only sign of life the watcher can pass on once the call has returned.
    """
    def _sink(ev: dict) -> None:
        try:
            compact = _compact_event(ev)
            if compact is None:
                return
            _append_activity(session, compact)
            if job_key and compact.get("t") == "tc":
                with _JOBS_LOCK:
                    entry = _JOBS.get(job_key)
                    if entry is not None:
                        entry["phase"] = compact.get("l") or compact.get("n") or ""
            q.put_nowait(compact)
        except Exception:
            counters["dropped"] = counters.get("dropped", 0) + 1
    return _sink


async def _report(ctx: Context | None, state: dict, payload: dict) -> None:
    """Send one event to the caller. A dead pipe must not fail the child's run."""
    if ctx is None:
        return
    state["sent"] = state.get("sent", 0) + 1
    try:
        await ctx.report_progress(
            progress=state["sent"],
            message=json.dumps(payload, separators=(",", ":")),
        )
    except Exception:
        pass


async def _forward_pending(ctx: Context | None, q: queue.Queue, state: dict) -> None:
    """Drain a bounded slice of the child's events to the caller."""
    for _ in range(_MAX_EVENTS_PER_TICK):
        try:
            ev = q.get_nowait()
        except queue.Empty:
            return
        if state.get("sent", 0) >= _MAX_EVENTS_PER_RUN:
            state["dropped"] = state.get("dropped", 0) + 1
            continue
        state["last_activity"] = time.monotonic()
        await _report(ctx, state, ev)


async def _maybe_heartbeat(ctx: Context | None, state: dict, started: float) -> None:
    """Say the child is alive when it has been silent long enough to look hung.

    Silence is the normal shape of a model turn — no tool call, nothing to forward —
    and a delegated turn can be minutes of it. Heartbeats are exempt from the per-run
    event ceiling: the one moment the caller most needs a sign of life is the run that
    already spent its budget on child steps.
    """
    now = time.monotonic()
    if now - state.get("last_activity", started) < _HEARTBEAT_SECS:
        return
    state["last_activity"] = now
    await _report(ctx, state, {"v": 1, "t": "hb", "s": int(now - started)})


def _caller_approval_mode(ctx: Context | None) -> str:
    """The approval mode the caller sent with this call, or "manual"."""
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        return "manual"
    mode = getattr(meta, "model_extra", None) or {}
    mode = str(mode.get(_APPROVAL_MODE_META) or "").strip().lower()
    return mode if mode in _APPROVAL_MODES else "manual"


def _caller_model(ctx: Context | None) -> str:
    """The model the caller runs on, from this call's _meta, else the spawn-time env.

    The env is only a fallback: it was frozen when this server started, and a
    mid-session model switch never reached it.
    """
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        meta = None
    model = str((getattr(meta, "model_extra", None) or {}).get(_CALLER_MODEL_META) or "").strip()
    if model:
        return model
    try:
        from mimir.client.config import DEFAULT_MODEL
    except ImportError:
        DEFAULT_MODEL = ""
    return os.environ.get("MIMIR_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL


def _refuse_model(model: str) -> str | None:
    """Why *model* cannot run this sub-agent, or None when it can."""
    from mimir.client.config.model_catalog import subagent_model_refusal
    return subagent_model_refusal(model, _SERVED_MODELS)


def _meta_value(ctx: Context | None, key: str):
    """One key of this call's request _meta, or None."""
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        return None
    return (getattr(meta, "model_extra", None) or {}).get(key)


def _caller_grantable(ctx: Context | None) -> dict[str, str]:
    """What the caller may hand to this child: ``{tool: owning server}``.

    The caller computes it — it is the only end that knows which servers the user has
    switched off, which mode it is in, and which tools are reserved to it. An absent
    or malformed table is read as "nothing may be granted", which costs an
    exploration nothing (its toolkit is named below by server, not by the caller) and
    keeps a caller that cannot vouch for a tool from handing it over.
    """
    raw = _meta_value(ctx, _GRANTABLE_TOOLS_META)
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if k and v}


def _caller_writers(ctx: Context | None) -> set[str]:
    """Which grantable tools change the tree, as the caller's registry reports them."""
    raw = _meta_value(ctx, _GRANTABLE_WRITERS_META)
    return {str(x) for x in raw} if isinstance(raw, list) else set()


def _caller_mode(ctx: Context | None) -> str:
    """The mode the caller is in ("agent", "plan", "ask"), or "" when it did not say."""
    return str(_meta_value(ctx, _CALLER_MODE_META) or "").strip().lower()


def _subagent_level(ctx: Context | None) -> str:
    """How far the user lets this child go; the weakest rung when unsaid or unknown."""
    level = str(_meta_value(ctx, _SUBAGENT_LEVEL_META) or "").strip().lower()
    return level if level in _SUBAGENT_LEVELS else _DEFAULT_SUBAGENT_LEVEL


def _level_allows(level: str, required: str) -> bool:
    return _SUBAGENT_LEVELS.index(level) >= _SUBAGENT_LEVELS.index(required)


def _sub_session_id() -> str:
    """A session id of this child's own, under the caller's session.

    ``<parent>/subagents/sub-xxxxxxxx`` — the parentage is the path, so the child's
    todo and scratchpad are its own, the front end never lists it (it enumerates the
    ``<id>.json`` files at the root of sessions/, and this writes only directories),
    and deleting the parent session takes its children with it.
    """
    child = f"sub-{uuid.uuid4().hex[:8]}"
    try:
        from state_paths import active_session_id
        parent = active_session_id()
    except Exception:
        parent = ""
    return f"{parent}/subagents/{child}" if parent else child


# ── Reaching the user from inside a sub-agent ─────────────────────────────────
# A child that needs an approval, or has a question, must be able to reach the person
# — otherwise the work stops on a step nobody was asked about. The route exists while
# the delegating call is open: this server may elicit on that call's session, and the
# client renders the card it already renders for any other server.
#
# Two rules the relay owes the user, both enforced here rather than hoped for:
#   * ONE AT A TIME. Several children can be working at once; several cards at once is
#     a pile nobody can answer in order. They queue on this lock, so a card is shown,
#     answered, and only then is the next one raised.
#   * WHO IS ASKING. Every card names the sub-agent and its task, because "allow this
#     command?" from an unnamed process is a question the user cannot weigh.
_ASK_LOCK = asyncio.Lock()

_APPROVAL_OPTIONS = [
    {"label": "Allow once", "description": "Run it this time."},
    {"label": "Allow for the session",
     "description": "Stop asking for this kind of call until the session ends."},
    {"label": "Refuse", "description": "Do not run it; the sub-agent carries on without."},
]


async def _ask_the_user(ctx: Context, header: str, question: str,
                        options: list[dict]) -> str:
    """Put one card to the user and return the label they chose, or "" .

    Queued behind every other sub-agent's card, so the answers cannot be attributed to
    the wrong question. A channel that fails is silence, not an answer: the caller
    reads "" as "they did not say", which every caller here treats as a refusal.
    """
    spec_questions = [{
        "question": question,
        "header": header[:24],
        "options": options,
        "multi_select": False,
    }]
    schema = {
        "type": "object",
        "title": header[:24] or "Sub-agent",
        "x_mimir": {"kind": "user_question", "questions": spec_questions},
        "properties": {"answers": {"type": "array", "items": {"type": "object"}}},
        "required": [],
    }
    async with _ASK_LOCK:
        try:
            result = await ctx.session.elicit_form(message=question, requestedSchema=schema)
        except Exception as exc:
            print(f"spawn_agent: could not reach the user: {exc}", file=sys.stderr)
            return ""
    if getattr(result, "action", "") != "accept":
        return ""
    content = getattr(result, "content", None) or {}
    try:
        answers = json.loads(content.get("answers") or "[]")
        selected = (answers[0] or {}).get("selected") or []
        return str(selected[0]) if selected else str((answers[0] or {}).get("other_text") or "")
    except Exception:
        return ""


def _install_user_channel(agent, ctx: Context, loop, task: str) -> None:
    """Route this child's approvals and questions to the user, through the caller.

    Installed only for a child whose delegating call is still open: the elicitation
    rides that call's session, and a detached child has none — it keeps running
    unattended and reports what its mode refused, as before.

    The child runs in its own thread with its own event loop, so each relay hops back
    onto the server's loop and blocks its own thread until the user answers. Blocking
    is correct here: the child has nothing to do until it has its answer.
    """
    label = task.strip().splitlines()[0][:60] if task.strip() else "a sub-agent"

    def _ask(header: str, question: str, options: list[dict]) -> str:
        future = asyncio.run_coroutine_threadsafe(
            _ask_the_user(ctx, header, question, options), loop)
        try:
            return future.result(timeout=SUBAGENT_HARD_CAP_SECS)
        except Exception:
            return ""

    def _approve_tool(tool_name: str, arguments: dict, max_attempts: int = 3):
        detail = json.dumps(arguments, ensure_ascii=False, default=str)[:400]
        answer = _ask(
            "Sub-agent",
            f"Sub-agent «{label}» wants to run {tool_name}.\n{detail}",
            _APPROVAL_OPTIONS,
        )
        if answer.startswith("Allow for the session"):
            agent.approvals.approve_scope(
                tool_name, agent.tool_owner.get(tool_name, "unknown"), arguments) \
                if hasattr(agent.approvals, "approve_scope") else None
            return True, "approved for this session"
        if answer.startswith("Allow"):
            return True, "approved once"
        return False, "the user refused"

    def _approve_paths(paths: list[str], tool_name: str, arguments: dict | None = None):
        shown = ", ".join(paths[:5]) + ("…" if len(paths) > 5 else "")
        answer = _ask(
            "Outside",
            f"Sub-agent «{label}» wants to reach outside the workspace: {shown}",
            _APPROVAL_OPTIONS,
        )
        if answer.startswith("Allow for the session"):
            return True, True
        return (answer.startswith("Allow"), False)

    def _question(questions: list) -> dict:
        answers = []
        for q in questions or []:
            chosen = _ask(
                str(q.get("header") or "Sub-agent"),
                f"Sub-agent «{label}» asks: {q.get('question') or ''}",
                list(q.get("options") or []),
            )
            answers.append({"header": q.get("header", ""),
                            "selected": [chosen] if chosen else [],
                            "other_text": None})
        return {"answers": answers}

    agent._request_tool_approval = _approve_tool
    agent._request_path_approval = _approve_paths
    agent._request_user_question = _question
    # It is no longer unattended: there is a person at the end of the channel, and the
    # mode gate must ask them rather than refuse on their behalf.
    agent.approvals.unattended = False


# ── A copy of the code, for an axis of its own ────────────────────────────────
# A sub-agent asked for one gets a git worktree: the same repository, its own branch
# and its own directory, so it can edit, build and measure without the run next door
# reading its files. Only worth it when several axes run at once — hence a rung the
# user sets, and an argument the caller passes.

# On the local disk, not in the state dir: a home under quota cannot hold a checkout
# plus everything a build writes. Deliberately NOT under the scratchpad either — that
# path is writable without approval (state_paths.standing_roots), and a copy placed
# there would take every edit the child makes out of the approval layer.
_WORKTREE_BASE = os.path.join(
    os.path.abspath(os.environ.get("TMPDIR") or "/tmp"),
    f"mimir-worktrees-{os.getuid()}",
)

_GIT_TIMEOUT_SECS = 120

# How many *finished* copies one session keeps, the copy being made not counted. A
# finished copy is worth keeping — it holds the build tree, which git never had and
# which cost the axis its compile — but nothing reclaims /tmp on its own, so this count
# is the bound: making the next copy drops the oldest beyond it.
SUBAGENT_WORKTREES_KEPT = 3


def _git(args: list[str], cwd: str) -> tuple[bool, str]:
    """Run one git command. Returns (ok, output) and never raises."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=_GIT_TIMEOUT_SECS,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    out = (done.stdout or "") + (done.stderr or "")
    return done.returncode == 0, out.strip()


def _repo_root() -> str:
    """The git repository this agent works in, or "" when there is none."""
    root = os.environ.get("MCP_FILES_ROOT") or os.getcwd()
    ok, out = _git(["rev-parse", "--show-toplevel"], root)
    return out if ok and out else ""


def _sibling_cards(session: str) -> tuple[str, list[dict]]:
    """The cards of every child of this one's parent session, newest first.

    The card is the registry: it is written for the panel anyway, and it is the only
    place that remembers where a finished child's copy was put. Best effort throughout
    — a session whose cards cannot be read reclaims nothing, which costs disk and
    breaks nothing.
    """
    parent = session.rsplit("/subagents/", 1)[0] if "/subagents/" in session else ""
    if not parent:
        return "", []
    try:
        from state_paths import state_dir
        base = os.path.join(state_dir(), "sessions", parent, "subagents")
        names = sorted(os.listdir(base))
    except Exception:
        return "", []
    # Newest first, by the card's own clock and then by when it was last written. The
    # timestamp has second resolution and a fan-out starts its children inside one
    # second, so on its own it would leave the order of a tie to the filesystem.
    rows = []
    for name in names:
        card_path = os.path.join(base, name, "subagent.json")
        try:
            with open(card_path, encoding="utf-8") as fh:
                card = json.load(fh)
            rows.append((card.get("started_at", ""), os.path.getmtime(card_path), card))
        except Exception:
            continue
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return base, [row[2] for row in rows]


def _card_is_live(card: dict) -> bool:
    """Whether a card that says "running" still belongs to a living process.

    A child runs in a thread of the server that spawned it, so its card is only ever
    finished by that process: one killed mid-run leaves a card saying "running" for
    good, and a copy that nothing would ever reclaim. The pid it recorded is the test.
    A card from before pids were recorded, or one belonging to another user's process,
    counts as live — erring towards keeping a copy rather than removing one that is
    still being written.
    """
    if card.get("state") != "running":
        return False
    pid = card.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _commit_copy(path: str, message: str) -> tuple[list[str], str]:
    """Stage and commit whatever the copy holds. Returns (files, diffstat).

    The one place work leaves a copy for good, and it is called from both ends: when
    the child finishes, and again before a copy is reclaimed. The second caller is why
    this exists as a function — a copy dropped with uncommitted work in it is work
    destroyed, and the reclaimer would have done it silently, three axes later.
    """
    _git(["add", "-A"], path)
    ok, changed = _git(["diff", "--cached", "--name-only"], path)
    files = [f for f in changed.splitlines() if f] if ok else []
    ok, stat = _git(["diff", "--cached", "--stat"], path)
    diffstat = stat if ok else ""
    if files:
        _git(["-c", "user.name=MIMIR", "-c", "user.email=mimir@localhost",
              "commit", "-m", message], path)
    return files, diffstat


def _drop_branch_if_empty(repo: str, branch: str, base: str) -> bool:
    """Delete *branch* when it is still exactly where it was cut from. Did it go?

    The test is against the commit recorded when the copy was made, not against the
    repository's HEAD: HEAD moves while a child works, and ``git branch -d`` would then
    refuse a branch that carries nothing simply because the ground shifted under it —
    leaving behind the litter this exists to remove. Emptiness established that way,
    ``-D`` is the honest way to say it: there is nothing left to protect.

    Without a recorded base — a card written before this — nothing is deleted. A branch
    kept for nothing costs a line in ``git branch``; one deleted wrongly costs an axis.
    """
    if not branch or not base:
        return False
    ok, tip = _git(["rev-parse", branch], repo)
    if not ok or tip != base:
        return False
    deleted, _ = _git(["branch", "-D", branch], repo)
    return deleted


def _marker_path(repo: str, child_key: str) -> str:
    """Where a copy's marker lives: beside the copy, never inside it.

    Inside, ``git add -A`` would commit it into the axis. Beside, it survives the copy
    being emptied and stays out of every diff.
    """
    return os.path.join(_WORKTREE_BASE, os.path.basename(repo), f"{child_key}.mimir.json")


def _write_marker(path: str, marker: dict) -> None:
    """Record who a copy belongs to, before it can be used. Best effort."""
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(marker, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _drop_copy(wt: dict, child_key: str = "") -> bool:
    """Commit whatever the copy holds, remove it, and drop a branch that carries nothing.

    The single way a copy ever goes, so the commit can never be skipped by one caller
    and not the other: a copy taken with uncommitted work in it is work destroyed.
    """
    path, repo = wt.get("path", ""), wt.get("repo", "")
    if not path or not repo:
        return False
    _commit_copy(path, f"sub-agent {os.path.basename(path)}: reclaimed unfinished work")
    removed, _ = _git(["worktree", "remove", "--force", path], repo)
    _git(["worktree", "prune"], repo)
    if not removed:
        return False
    _drop_branch_if_empty(repo, wt.get("branch") or "", wt.get("base") or "")
    with contextlib.suppress(OSError):
        os.remove(_marker_path(repo, child_key or os.path.basename(path)))
    return True


def _sweep_orphan_copies(repo: str) -> None:
    """Drop copies of this repository that nothing remembers any more.

    The cards reclaim a session's own copies, but they are stored *with* the session:
    deleting a conversation takes the registry away and leaves the copies on disk, where
    the per-session rule can no longer reach them. What survives that is the copy's
    marker, written beside it before it could be used.

    A copy is an orphan when its owner is gone *and* its card is gone — both, because
    either alone is a copy that is simply between states: a child still running holds
    its pid, and a card is written a moment after the copy is made. One with no marker
    at all predates this and is judged by git alone: a directory git does not register
    is not a worktree anybody is using.
    """
    base = os.path.join(_WORKTREE_BASE, os.path.basename(repo))
    if not os.path.isdir(base):
        return
    ok, listed = _git(["worktree", "list", "--porcelain"], repo)
    registered = {line.split(" ", 1)[1] for line in listed.splitlines()
                  if ok and line.startswith("worktree ")}
    try:
        from state_paths import state_dir
        sessions = os.path.join(state_dir(), "sessions")
    except Exception:
        return
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        marker = _marker_path(repo, name)
        try:
            with open(marker, encoding="utf-8") as fh:
                owner = json.load(fh)
        except (OSError, json.JSONDecodeError):
            if os.path.realpath(path) not in {os.path.realpath(p) for p in registered}:
                shutil.rmtree(path, ignore_errors=True)
            continue
        if _card_is_live({"state": "running", "pid": owner.get("pid")}):
            continue
        if os.path.isdir(os.path.join(sessions, owner.get("session", ""))):
            continue          # its session still holds its card; the per-session rule owns it
        _drop_copy(owner, name)


def _reclaim_worktrees(session: str, keep: int) -> None:
    """Drop this session's finished copies beyond the *keep* most recent.

    Called when the next copy is about to be made, which is the moment the disk is
    known to be wanted — so a session holds *keep* finished copies plus the one being
    made. A child still running is never counted, and a copy git refuses to remove is
    left where it is.

    Only copies that are actually on disk take a place: cards from before copies were
    kept name a directory that is long gone, and counting those would push out a copy
    that exists.

    "Still running" is read off the pid on the card, not off the word: a server killed
    mid-run leaves its card saying "running" for ever, and taking that at face value
    would exempt its copy from reclamation forever after.
    """
    _, cards = _sibling_cards(session)
    copies = [w for w in (c.get("workspace") or {} for c in cards if not _card_is_live(c))
              if w.get("path") and w.get("repo") and os.path.isdir(w["path"])]
    for wt in copies[keep:]:
        _drop_copy(wt)


def _create_worktree(child_key: str, session: str = "") -> tuple[dict | None, str]:
    """Add a worktree for this child. Returns (descriptor, error message)."""
    repo = _repo_root()
    if not repo:
        return None, (
            "a sub-agent that writes needs a copy of the repository, and this workspace "
            "is not a git repository — so there is none to make. Delegate the reading "
            "and do the writing yourself."
        )
    branch = f"mimir/{child_key}"
    path = os.path.join(_WORKTREE_BASE, os.path.basename(repo), child_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Before adding one, not after: the copies this session is done with are what makes
    # room for this one, and reclaiming them later would mean nothing ever did.
    if session:
        _reclaim_worktrees(session, SUBAGENT_WORKTREES_KEPT)
    # A copy whose directory went without git being told still holds its registration,
    # and those accumulate in .git/worktrees/ where only this ever looks.
    _git(["worktree", "prune"], repo)
    # And the ones no session remembers any more: a deleted conversation takes its cards
    # with it, and the per-session rule above can no longer see what it left behind.
    _sweep_orphan_copies(repo)
    # From HEAD, not from the working tree: a worktree cannot carry uncommitted changes,
    # and saying so beats a child hunting for code it cannot find.
    ok, out = _git(["worktree", "add", "-b", branch, path, "HEAD"], repo)
    if not ok:
        return None, f"could not create the working copy: {out.splitlines()[-1] if out else 'git failed'}"
    # The commit it was cut from, kept as long as the copy is: it is what later tells a
    # branch carrying an axis from one carrying nothing, whatever the repository's HEAD
    # has moved on to by then.
    based, base = _git(["rev-parse", "HEAD"], path)
    descriptor = {"repo": repo, "branch": branch, "path": path,
                  "base": base if based else ""}
    # Written before the copy is handed over, so no sweep can ever find it unowned: the
    # card that says the same thing lands a moment later, in the child's own thread.
    _write_marker(_marker_path(repo, child_key),
                  {**descriptor, "session": session, "pid": os.getpid()})
    return descriptor, ""


def _finish_worktree(wt: dict, task: str, succeeded: bool) -> dict:
    """Commit the axis and keep the copy.

    The branch is what carries the *code*: a worktree shares the repository's object
    store, so a commit survives the directory. Which is why the commit is made here
    rather than asked of the child — a copy removed with uncommitted work in it is work
    destroyed, and that must not depend on the model remembering.

    But the copy carries what git never had: the build tree. Dropping it here threw
    away the compile the axis had just paid for, and left behind a branch that, for
    every child that wrote nothing, pointed at HEAD and said nothing. So the directory
    stays — bounded by :func:`_reclaim_worktrees`, which drops the oldest when the next
    child needs room — and it is the empty branch that goes instead.
    """
    path, branch, repo = wt["path"], wt["branch"], wt["repo"]
    base = wt.get("base", "")
    report = {"branch": branch, "path": path, "repo": repo, "base": base,
              "files_changed": [], "diffstat": "", "kept": True}

    if not succeeded:
        # Staged, not committed: a run that stopped early is continued in its copy, and
        # a commit taken mid-edit would have to be undone before that could start.
        _git(["add", "-A"], path)
        ok, changed = _git(["diff", "--cached", "--name-only"], path)
        report["files_changed"] = [f for f in changed.splitlines() if f] if ok else []
        ok, stat = _git(["diff", "--cached", "--stat"], path)
        report["diffstat"] = stat if ok else ""
        report["note"] = (
            f"The run did not finish. Its copy is at {path}, with its uncommitted state "
            f"left as it was — that state is the only place that work exists.")
        return report

    files, diffstat = _commit_copy(
        path, f"sub-agent {os.path.basename(path)}: {task[:120]}")
    report["files_changed"], report["diffstat"] = files, diffstat
    if files:
        report["committed"] = True
        report["note"] = (
            f"Its work is committed on {branch}, and the copy is still at {path}. Git "
            f"refuses to check that branch out anywhere else while the copy holds it, "
            f"so bring the work over with merge or cherry-pick, not checkout.")
        return report

    # Nothing written: the branch is still at the commit it was cut from and carries
    # nothing. Detaching first is what lets it go — git will not delete a branch a
    # worktree has checked out, and the copy itself is worth keeping either way. If
    # either step fails the branch is still there, and the report has to keep saying so.
    _git(["checkout", "--detach"], path)
    if _drop_branch_if_empty(repo, branch, base):
        report["branch"] = ""
        report["note"] = f"It changed no file, so it has no branch. Its copy is at {path}."
    else:
        report["note"] = (f"It changed no file. Its copy is at {path}, and {branch} is "
                          f"still where it was cut from.")
    return report


# ── Detached sub-agents ───────────────────────────────────────────────────────
# A sub-agent the caller asked to run in the background outlives its own tool call.
# The call returns a ``background_job`` descriptor, the client's watcher polls the
# job tool below, and the caller is resumed with the answer when the child is done —
# the same machinery a detached shell command uses, so nothing in the client knows
# this kind of job from that one.
#
# The registry is this process's memory of those children. Written from the tool's
# thread and read from the server loop, so every access takes the lock.

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

# Enough to answer for the children of one session; past it the oldest finished entry
# goes. A run is never dropped while it is still running.
_JOBS_MAX = 64


def _job_descriptor(job_key: str) -> dict:
    """The handle the client's watcher polls.

    Data, not a tool name in loop code: the client reads ``status_op`` / ``summary_op``
    off it and polls them generically, which is how a detached run gets watched
    without the caller spending a model call per poll. ``summary_op`` is also what the
    caller may read *while* the child works — the dispatch guard refuses the status
    question, never the progress one.
    """
    return {
        "server": "agent",
        "kind": "sub-agent",
        "job_key": job_key,
        "status_op": {"tool": "subagent_job", "args": {"job_key": job_key}},
        "summary_op": {"tool": "subagent_job",
                       "args": {"op": "result", "job_key": job_key}},
    }


def _register_job(job_key: str, entry: dict) -> None:
    with _JOBS_LOCK:
        if len(_JOBS) >= _JOBS_MAX:
            for key, old in sorted(_JOBS.items(), key=lambda kv: kv[1].get("started", 0)):
                if old.get("state") != "running":
                    _JOBS.pop(key, None)
                    break
        _JOBS[job_key] = entry


def _job_snapshot(job_key: str) -> dict | None:
    with _JOBS_LOCK:
        entry = _JOBS.get(job_key)
        return dict(entry) if entry else None


def _settle_job(job_key: str, state: str, result: dict | None) -> None:
    with _JOBS_LOCK:
        entry = _JOBS.get(job_key)
        if entry is not None:
            entry["state"] = state
            entry["result"] = result
            entry["ended"] = time.time()


def _job_state(entry: dict) -> str:
    """The state the watcher acts on: running, done, crashed — or unknown past the budget.

    A child that overruns keeps going in its own thread, so "unknown" is the honest
    answer: the run was let go, and what it had done is still reported.
    """
    if entry.get("state") != "running":
        return entry["state"]
    if time.time() - entry.get("started", 0) > entry.get("budget", SUBAGENT_HARD_CAP_SECS):
        return "unknown"
    return "running"


# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[DELEGATE, BACKGROUNDABLE],
    # Reversible, hence not approval-gated: a card in front of every exploration is a
    # card in front of the behaviour this tool exists to make cheap. A writing child
    # is gated by its own approval layer — and in a read-only mode there is nothing
    # writing to grant it, because the caller's own tool surface has none.
    reversibility=REVERSIBLE,
    # BACKGROUNDABLE: a child the caller detached returns a background_job handle
    # instead of an answer, and the client's watcher resumes the caller when it lands —
    # the same machinery a detached shell command uses.
    # Deliberately above the tool's own SUBAGENT_HARD_CAP_SECS, so the inner cap fires
    # first and returns what the child had instead of the dispatcher killing the call
    # with nothing to show.
    timeout_secs=SUBAGENT_HARD_CAP_SECS + 60,
    label="Sub-agent: {task}",
    read_only=False,
))
async def spawn_agent(
    task: str,
    context: str = "",
    tools: Annotated[list[str], Field(description=_TOOLS_DESCRIPTION)] = [],
    max_steps: int = 30,
    time_budget_secs: int = SUBAGENT_DEFAULT_BUDGET_SECS,
    background: bool = False,
    model: Annotated[str, Field(description=_MODEL_DESCRIPTION)] = "",
    ctx: Context | None = None,
) -> dict:
    """Hand a self-contained sub-task to a fresh sub-agent, and get its answer back.

    The sub-agent has its own conversation history and execution context, so what it
    reads never enters yours — only its answer does. Emit several calls **in the same
    response** and they run concurrently; that fan-out is the point of the tool, and a
    broad sweep splits naturally into independent questions.

    Args:
        task:       The complete, self-contained task or question for the sub-agent.
                    It sees none of your conversation, so say everything it needs.
        context:    Optional extra context (findings so far, constraints) prepended
                    to the task.
        max_steps:  Max tool-call steps the sub-agent may take (default 30).
        time_budget_secs:
                    Wall time for this run (default 600, max 1140). The sub-agent is
                    told its budget and is asked to hand over before it runs out.
        background: True detaches the sub-agent: this call returns a handle at once and
                    you are resumed with its answer when it finishes, so you can work
                    on something else meanwhile. Use it for a long piece of work you
                    are not waiting on — several axes explored at the same time, a
                    build-and-measure cycle — and leave it false when its answer is
                    your next step. A detached child's steps are not shown in your
                    rows; ask for its result if you need what it has so far.
    Returns:
        On success (the sub-agent ran to a result):
            {"status": "ok", "answer": "…", "completed": bool, "files_read": [...],
             "files_written": [...], "model": "…"}
            - ``answer``        — the sub-agent's final answer (primary payload).
            - ``completed``     — False when the sub-agent ran out of steps or reported
                                  the task incomplete (the answer is still informative).
            - ``files_read``    — what the child actually opened, so the evidence it
                                  gathered is on the record for the caller too.
            - ``files_written`` — workspace files the sub-agent modified (empty when it
                                  was granted no writing tool); lets a parent
                                  coordinating concurrent sub-agents detect overlapping
                                  edits.
            - ``blocked_by_mode`` — actions the sub-agent skipped because the user's
                                  approval mode does not allow them (it cannot ask).
                                  Each is ``{"action", "needs"}``; when non-empty,
                                  ``completed`` is False. Tell the user which mode
                                  would let them run rather than retrying.
            - ``model``         — the model the sub-agent ran on.
            - ``tools``         — the tools it actually had.
            - ``session``       — its own session id. Its todo list and scratchpad live
                                  there, and they outlive the call: hand it to the next
                                  sub-agent instead of starting the axis again.
            - ``workspace``     — empty when it only read, in your own files. When it
                                  was given a writing or running tool it worked in a
                                  COPY of the repository: the ``branch`` its work was
                                  committed to, the ``path`` of that copy, the files it
                                  changed and a ``diffstat``. ``kept`` says whether the
                                  copy is still on disk — it is only when the run did
                                  not finish. Read the branch, compare the axes, and
                                  merge the one you keep yourself.
        On failure (the sub-agent crashed or ran out of time):
            {"status": "error", "error": "…", "answer": "<partial>", ...}
            — distinct ``status`` so the orchestrator can branch on failure without
              string-matching the answer text.
    """
    grantable = _caller_grantable(ctx)
    requested = [str(t).strip() for t in (tools or []) if str(t).strip()]
    unavailable = [t for t in requested if t not in grantable]
    if unavailable:
        offer = ", ".join(sorted(grantable)) if grantable else "(none)"
        return err(
            f"cannot grant {', '.join(unavailable)}: not available to you right now, or "
            f"reserved to you (planning, delegation, asking the user, writing memory, "
            f"cluster submission). What you may grant: {offer}. Leave `tools` empty for a "
            f"read-only exploration.",
            answer="", completed=False, files_read=[], files_written=[],
        )
    level = _subagent_level(ctx)
    budget = max(SUBAGENT_MIN_BUDGET_SECS,
                 min(int(time_budget_secs or SUBAGENT_DEFAULT_BUDGET_SECS),
                     SUBAGENT_HARD_CAP_SECS))

    caller_model = _caller_model(ctx)
    model = (model or "").strip()
    if model and model != caller_model:
        refusal = _refuse_model(model)
        if refusal:
            return err(refusal, answer="", completed=False, files_read=[], files_written=[])
    else:
        model = caller_model

    _silence_stdout_once()
    approval_mode = _caller_approval_mode(ctx)
    session = _sub_session_id()
    child_key = session.rsplit("/", 1)[-1]
    job_key = child_key if background else ""

    # A child that was granted anything writing or executing gets a copy of the
    # repository — always, and not because it asked. Several of them editing one tree
    # overwrite each other in silence, and no instruction to the model prevents that;
    # a separate worktree does, by construction. A reading child stays in these files,
    # where it can do no harm.
    worktree: dict | None = None
    if set(requested) & _caller_writers(ctx):
        worktree, refusal = _create_worktree(child_key, session)
        if worktree is None:
            return err(refusal, answer="", completed=False, files_read=[], files_written=[])

    # The child runs in a dedicated thread with its own event loop: its MCP exit
    # stack must be opened and closed in one task on one loop, and the hard cap
    # below must not become a task cancellation in the middle of a stdio_client.
    # This tool stays async so waiting on that thread never blocks this server's
    # loop — that is what makes a fan-out of several calls actually concurrent.
    events: queue.Queue = queue.Queue(maxsize=_CHILD_QUEUE_MAX)
    counters: dict = {"dropped": 0}
    state: dict = {"sent": 0, "dropped": 0}
    future: Future[dict] = Future()
    # The running child, published here by the thread. A run that overruns its budget
    # keeps going in its own thread and its result never arrives — without this the
    # caller was told only that the time was up. Read defensively: whatever is in it
    # is being written by another thread.
    child: dict = {}

    def _thread_main() -> None:
        try:
            result = asyncio.run(_run_sub_agent(
                task, context, requested, grantable, max_steps, approval_mode,
                on_event=_make_child_sink(events, counters, job_key, session),
                model=model, session=session, budget=budget, child=child,
                level=level, worktree=worktree,
                # A detached child gets no channel: its call has returned, and there
                # is no session left to raise a card on.
                ask_ctx=None if background else ctx,
                ask_loop=None if background else loop,
            ))
            future.set_result(result)
            if job_key:
                _settle_job(job_key, "done" if not result.get("error") else "crashed",
                            result)
        except Exception as exc:
            future.set_exception(exc)
            if job_key:
                _settle_job(job_key, "crashed", {"answer": "", "completed": False,
                                                 "error": str(exc)})

    if job_key:
        # Registered before the thread starts: the caller is handed the key on the next
        # line, and a watcher that polled a key this process did not yet know would read
        # it as a job that never existed.
        _register_job(job_key, {
            "state": "running", "started": time.time(), "budget": budget,
            "task": task, "session": session, "model": model, "tools": requested,
            "phase": "", "result": None, "child": child,
        })

    loop = asyncio.get_running_loop()
    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()

    if job_key:
        return ok({
            "answer": "",
            "completed": False,
            "files_read": [], "files_written": [], "blocked_by_mode": [],
            "model": model, "tools": requested, "session": session,
            "background_job": _job_descriptor(job_key),
            "workspace": {"branch": worktree["branch"], "path": worktree["path"]}
                         if worktree else {},
            "note": "Started in the background; this call returns before the sub-agent "
                    "finishes. You are resumed with its answer when it does.",
        })

    started = time.monotonic()
    state["last_activity"] = started
    deadline = loop.time() + budget
    while not future.done() and loop.time() < deadline:
        await _forward_pending(ctx, events, state)
        await _maybe_heartbeat(ctx, state, started)
        await asyncio.sleep(_POLL_SECS)
    # One last drain, before the timeout branch too: the child's final tool result
    # is what tells the caller where a run that ran out of time actually stopped.
    await _forward_pending(ctx, events, state)
    dropped = counters.get("dropped", 0) + state.get("dropped", 0)
    if dropped:
        await _report(ctx, state, {"v": 1, "t": "end", "dropped": dropped})

    if not future.done():
        # Out of time, with the child still running in its own thread: its answer will
        # never arrive here, so stop it before reading it. Then hand back what can be
        # read off it instead of a bare failure line — where it got to is what lets
        # the caller carry the axis on.
        _abandon_child(child)
        partial = _partial_handoff(child)
        return err(
            f"sub-agent ran out of its {budget}s budget before handing over, and was "
            f"stopped. Its work so far is below. Do not spawn the same task again as "
            f"it stands — a full budget has just been spent on it, and nothing about "
            f"it has changed. Either carry it on yourself, or delegate one narrower "
            f"step of it with a larger time_budget_secs, saying in the task what this "
            f"run already established and which approaches not to retry.",
            answer=partial.pop("answer", ""), completed=False, model=model,
            tools=requested, session=session, workspace=child.get("workspace", {}),
            **partial,
        )
    try:
        result = future.result(timeout=0)
    except Exception as exc:
        # The child's thread crashed — a genuine failure, not an "ok" result.
        return err(
            f"sub-agent failed: {exc}",
            answer="", completed=False, files_read=[], files_written=[],
            model=model, tools=requested, session=session,
            workspace=child.get("workspace", {}),
        )

    if result.get("error"):
        # The sub-agent's own run() raised — surface as an error, keep the partial answer.
        return err(
            f"sub-agent crashed: {result['error']}",
            answer=result.get("answer", ""),
            completed=False,
            files_read=result.get("files_read", []),
            files_written=result.get("files_written", []),
            blocked_by_mode=result.get("blocked_by_mode", []),
            model=model, tools=requested, session=session,
            workspace=result.get("workspace", {}),
        )
    if not str(result.get("answer") or "").strip():
        # The answer IS the payload — a blank one carries nothing back, whatever the
        # child did. Reported "ok"/completed, it read as a sub-agent that ran and found
        # nothing to say, and the parent dropped delegation for the rest of the run.
        return err(
            "sub-agent returned no answer: nothing was delegated back. Re-issue the "
            "call with a narrower, self-contained question, or do the work yourself.",
            answer="",
            completed=False,
            files_read=result.get("files_read", []),
            files_written=result.get("files_written", []),
            model=model, tools=requested, session=session,
        )
    return ok({
        "answer": result["answer"],
        "completed": result["completed"],
        "files_read": result["files_read"],
        "files_written": result["files_written"],
        "blocked_by_mode": result["blocked_by_mode"],
        "model": model,
        "tools": requested,
        "session": session,
        "workspace": result.get("workspace", {}),
    })


@mcp.tool(**tool_caps(
    # MAIN_ONLY: these are the caller's own handles. A sub-agent has no business
    # reading its siblings, and cannot start one of its own anyway.
    caps=[MAIN_ONLY],
    read_only=True,
    label="Sub-agent job: {op}",
))
def subagent_job(
    op: Annotated[str, Field(description="status (default) | result | list")] = "status",
    job_key: str = "",
) -> dict:
    """Read a sub-agent you started in the background: its state, its answer, or the list.

    Read-only, so it needs no approval.

        status  running | done | crashed | unknown, with the elapsed time and what the
                sub-agent is currently doing.
        result  everything the sub-agent returned — the same payload a blocking call
                gives back. While it still runs, this is what it has done so far.
        list    every sub-agent of this session, newest first.

    You are resumed automatically when a background sub-agent finishes, so do not poll
    its status: ask for its ``result`` if you need what it has so far, and otherwise
    get on with other work.

    Args:
        job_key: The handle spawn_agent(background=True) returned.
    """
    op = (op or "status").strip().lower()
    if op == "list":
        with _JOBS_LOCK:
            jobs = [
                {"job_key": key, "state": _job_state(entry), "task": entry.get("task", ""),
                 "session": entry.get("session", ""),
                 "elapsed_secs": int(time.time() - entry.get("started", 0))}
                for key, entry in _JOBS.items()
            ]
        jobs.sort(key=lambda j: j["elapsed_secs"])
        return ok({"jobs": jobs, "count": len(jobs)})

    entry = _job_snapshot(job_key)
    if entry is None:
        return err(f"unknown sub-agent job {job_key!r}: it was never started here, or "
                   f"this session has since forgotten it. Use op='list' to see the "
                   f"ones it still holds.")
    state = _job_state(entry)
    base = {
        "state": state,
        "job_key": job_key,
        "task": entry.get("task", ""),
        "session": entry.get("session", ""),
        "model": entry.get("model", ""),
        "tools": entry.get("tools", []),
        "elapsed_secs": int((entry.get("ended") or time.time()) - entry.get("started", 0)),
        # What it is doing, in the words of its own last tool row. The watcher passes
        # this on as the job's phase, which is the only sign of life a detached child
        # has once its call has returned.
        "phase": entry.get("phase", ""),
    }
    if op != "result":
        return ok(base)
    result = entry.get("result")
    if result is None:
        # Still working: what it has touched so far is real, its answer is not yet.
        return ok({**base, **_partial_handoff(entry.get("child") or {}),
                   "completed": False})
    return ok({**base, **result, "completed": bool(result.get("completed"))})


async def _run_sub_agent(
    task: str,
    context: str,
    requested: list[str],
    grantable: dict[str, str],
    max_steps: int,
    approval_mode: str = "manual",
    on_event: Callable[[dict], None] | None = None,
    model: str = "",
    session: str = "",
    budget: int = SUBAGENT_DEFAULT_BUDGET_SECS,
    child: dict | None = None,
    level: str = _DEFAULT_SUBAGENT_LEVEL,
    worktree: dict | None = None,
    ask_ctx: Context | None = None,
    ask_loop=None,
) -> dict:
    """Async implementation: create MimirAgent, wire tools, run query.

    Returns ``{"answer", "completed", "files_read", "files_written", "error"}``;
    ``error`` is None on a clean run (the task may still be incomplete — see
    ``completed``).
    """
    from mimir.client.agent_core import MimirAgent

    # The tool already settled the model: the caller's, or one it named that the
    # endpoint serves. Its temperature, window and reasoning profile resolve per model.
    agent = MimirAgent(model=model or _caller_model(None))
    # Its own session, so its todo list and its scratchpad are its own. Read by every
    # server it is about to start (server_manager merges this into their environment),
    # never by the servers already running for the caller.
    agent.session_id = session
    agent.server_env = {"MIMIR_SESSION_ID": session} if session else {}
    if worktree:
        # Its own root: the servers it starts read the copy, and the client-side gates
        # judge "inside the workspace" against it too.
        agent.workspace_root = worktree["path"]
        agent.server_env.update({
            "MCP_FILES_ROOT": worktree["path"],
            "SEARCH_ROOT": worktree["path"],
            # The proxy store stays the caller's: a sealed reference and a benchmark
            # suite are keyed by their own names, so the axes share them and only their
            # optimisation sessions (keyed by proxy name) are separate.
            "MIMIR_PROXY_BENCH_DIR": os.path.join(worktree["repo"], "proxy_bench"),
        })
    if child is not None:
        child["agent"] = agent
        # Also the copy it works in: a run that overruns its budget is read through
        # _partial_handoff, and that is where the paths it reports get their names.
        child["worktree"] = worktree
    record = {
        "session": session, "task": task, "tools": list(requested),
        "model": agent.model, "budget_secs": budget, "max_steps": max_steps,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "running",
        # Whose "running" this is. A card is only ever finished by the process that
        # wrote it, so one killed mid-run says "running" for ever; the pid is how a
        # later reclaim tells that from a child that really is still working.
        "pid": os.getpid(),
    }
    _write_record(session, record)
    result: dict | None = None
    try:
        result = await _drive_sub_agent(
            agent, task, context, requested, grantable, max_steps, on_event,
            approval_mode=approval_mode, budget=budget, level=level, worktree=worktree,
            ask_ctx=ask_ctx, ask_loop=ask_loop)
        return result
    finally:
        if worktree is not None:
            # Committed and dropped on a clean run, kept when the run did not finish:
            # an unfinished axis has its state nowhere else.
            clean = bool(result) and not result.get("error")
            record["workspace"] = _finish_worktree(worktree, task, clean)
            if result is not None:
                result["workspace"] = record["workspace"]
            if child is not None:
                # Also where a crash can read it: the thread's exception never carries
                # a result, and a kept copy is precisely what that caller must be told.
                child["workspace"] = record["workspace"]
        _write_record(session, {
            **record,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # No result at all means the run raised. A run whose caller timed out is
            # neither: _abandon_child marked it on the way out, and its answer — clean
            # or not — was never delivered, so it never counts as completed.
            "state": _final_state(child, result),
            "completed": bool((result or {}).get("completed"))
                         and not (child or {}).get("abandoned"),
            # Whole, not clipped: this file is the only copy that outlives the
            # run, and the panel is the only place it is ever read. The list view
            # cuts it for its own payload (session_store.list_subagents); cutting
            # here would cut it everywhere, for good.
            "answer": (result or {}).get("answer") or "",
            "files_written": (result or {}).get("files_written", []),
        })
        # Close the child's MCP stdio sessions HERE, in the very task that opened
        # them. Left to asyncio.run's shutdown_asyncgens, each stdio_client would be
        # closed from another task and blow up on anyio's cancel-scope check (and
        # leak one server subprocess per connected server).
        try:
            await agent.cleanup()
        except Exception as exc:  # teardown races must not mask the child's answer
            print(f"spawn_agent: sub-agent cleanup warning: {exc}", file=sys.stderr)


async def _drive_sub_agent(
    agent,
    task: str,
    context: str,
    requested: list[str],
    grantable: dict[str, str],
    max_steps: int,
    on_event: Callable[[dict], None] | None = None,
    approval_mode: str = "manual",
    budget: int = SUBAGENT_DEFAULT_BUDGET_SECS,
    level: str = _DEFAULT_SUBAGENT_LEVEL,
    worktree: dict | None = None,
    ask_ctx: Context | None = None,
    ask_loop=None,
) -> dict:
    """Configure the child agent, run it, and report what it did."""
    # _run_sub_agent already fixed sys.path if needed, so a plain import is safe here.
    from mimir.client.extensions import all_servers
    from mimir.client.context.capabilities import (
        PLAN_BLOCKED, PLAN_READONLY, explorer_servers, has_cap,
    )

    # Sub-agents don't reason out loud: set the rung, not just the run() flag, since
    # the loop re-reads the agent's depth every step (the user steering the parent's
    # thinking must not leak into a child run).
    agent.set_thinking_depth(0)
    # A conclusion is a paragraph, and a tool-call step is shorter still. Uncapped,
    # one step may claim the whole answer reserve — measured at ~40k tokens, i.e.
    # several minutes of generation during which the child emits nothing at all.
    agent.max_answer_tokens = SUBAGENT_ANSWER_TOKENS
    # The user's approval mode, not the default: the child acts for the same user. It
    # has no one to ask (its stdin is this server's JSON-RPC pipe), so what the mode
    # does not cover is refused without a prompt and reported back as blocked.
    agent.set_approval_mode(approval_mode)
    agent.approvals.unattended = True
    if ask_ctx is not None and ask_loop is not None:
        # There is a person at the end of this call: its approvals and its questions go
        # to them, queued behind any other child's and saying which child is asking.
        _install_user_channel(agent, ask_ctx, ask_loop, task)
    # The child writes the shared path allowlist too; starting from what is already
    # there keeps its writes from erasing the caller's grants.
    agent.approvals._allowed_paths.update(approved_roots())

    _servers = all_servers()
    if requested:
        # Only the servers the granted tools live in. The caller sent the owner of each
        # name, because the child connects servers and a tool cannot be started on its
        # own — and every server started costs a subprocess before the first step.
        wanted = {grantable[name] for name in requested if name in grantable}
        servers_to_connect = {k: v for k, v in _servers.items() if k in wanted}
    else:
        # No tools named: reconnaissance. A cost filter, not the guarantee — what keeps
        # this child read-only is the mode it runs in.
        servers_to_connect = {k: v for k, v in _servers.items() if k in explorer_servers()}

    for name, script in servers_to_connect.items():
        try:
            await agent.connect_server(name, script)
        except Exception as exc:
            print(f"spawn_agent: could not connect server '{name}': {exc}", file=sys.stderr)

    if requested:
        _prune_tools(agent, requested)

    # Seed classification from this sub-agent's own (subset) registry — keeps
    # approval/plan-block/caching correct and per-agent (the sub-agent connects
    # fewer servers than the parent, and keeps fewer tools still).
    agent.seed_classification_from_caps()

    agent._spawn_mode = True  # marker flag (not currently used by loop)

    # What the child was given decides how it runs. Nothing that writes or executes
    # among its tools means nothing to gate: it explores, in a read-only mode. One such
    # tool and it works, which is the only way a granted shell can actually run a build.
    # Set on the agent as well as passed to run(), so the loop's mode tracking starts
    # where it ends and reads no switch on the first step.
    working = _level_allows(level, "parallel") and any(
        has_cap(name, PLAN_BLOCKED, agent.tool_caps)
        or has_cap(name, PLAN_READONLY, agent.tool_caps)
        for name in requested
    )
    mode = "agent" if working else _READONLY_CHILD_MODE
    agent.set_mode(mode)
    # Set here rather than with the rest of the child's setup: how much window the
    # child may fill follows from what it was given, and that is not known until its
    # tools have been classified just above.
    agent.max_context_tokens = (
        SUBAGENT_CONTEXT_TOKENS_WORKING if working else SUBAGENT_CONTEXT_TOKENS_EXPLORE
    )

    brief = (
        _WORK_BRIEF.format(budget=_human_budget(budget), steps=max_steps)
        if working else _EXPLORE_BRIEF
    )
    if worktree:
        brief += _COPY_BRIEF.format(path=worktree["path"], branch=worktree["branch"])
    query = "\n\n---\n\n".join(p for p in (brief, context.rstrip(), task) if p)

    print(f"⟳ sub-agent starting ({mode}): {task[:80]}{'…' if len(task) > 80 else ''}",
          file=sys.stderr)
    error: str | None = None
    try:
        answer = await agent.run(
            query=query,
            max_steps=max_steps,
            mode=mode,
            streaming=False,   # sub-agents don't stream tokens to the UI
            thinking=False,
            # Binding a sink does double duty: the caller gets to see what the child
            # is doing, and the child's engine stops falling back to printing every
            # event — which here means printing into the JSON-RPC pipe.
            event_callback=on_event,
        )
    except Exception as exc:
        answer = f"[sub-agent error] {exc}"
        error = str(exc)
    print(f"✓ sub-agent finished: {task[:60]}{'…' if len(task) > 60 else ''}",
          file=sys.stderr)

    # What the child touched this run (recorded by _update_carry_context). `files_read`
    # goes back so the caller can record the evidence a delegated sweep produced.
    _carry = agent._carry_context
    files_read = _repo_relative(sorted(_carry.get("read_files", set())), worktree)
    files_written = _repo_relative(
        sorted(_carry.get("last_query_written_files", set())), worktree)
    # "completed" = the task itself finished; distinct from the run succeeding.
    # finalize_incomplete_answer owns its headlines (is_incomplete_answer matches
    # them); the hard step limit yields "Reached the maximum number of steps…".
    from mimir.client.guardrails.workflow import is_incomplete_answer

    blocked_by_mode = list(agent.approvals.mode_blocked)
    completed = error is None and not blocked_by_mode and not (
        not answer.strip()
        or is_incomplete_answer(answer)
        or answer.startswith("Reached the maximum number of steps")
    )
    return {
        "answer": answer,
        "completed": completed,
        "files_read": files_read,
        "files_written": files_written,
        "blocked_by_mode": blocked_by_mode,
        "error": error,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
