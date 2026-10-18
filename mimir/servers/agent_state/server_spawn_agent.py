"""MCP Spawn-Agent Server: delegate a sub-task to a fresh MimirAgent and return its answer.

Async, so several calls in one step run concurrently; the child's tool calls stream back
as progress. It gets only the tools the caller grants (and their servers), on the
caller's model unless another served one is named.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Annotated, Any, Callable

from . import child_jobs
from .._shared.approved_roots import approved_roots
from .._shared.atomic import write_json
from .._shared.capabilities import (
    BACKGROUNDABLE,
    DELEGATE,
    MAIN_ONLY,
    PLAN_READONLY,
    REVERSIBLE,
    tool_caps,
)
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from .._shared.responses import err, ok
from . import spawn_limits
from . import spawn_user_channel
from . import spawn_worktree
from . import spawn_jobs
from . import spawn_output

mcp = FastMCP("spawn_agent")

# The read-only mode a child without any writing tool runs in (one of the client's
# config.models.READONLY_MODES). "ask" rather than "plan": the child answers a
# question, it does not draft a plan.
_READONLY_CHILD_MODE = "ask"


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

# How far the user lets a sub-agent go (client config.settings.SUBAGENT_LEVELS,
# config.models.SUBAGENT_LEVEL_META). A rung hands something over, so an absent or
# unknown one is read as the weakest: a child that only reads.
_SUBAGENT_LEVELS = ("explore", "build")
_DEFAULT_SUBAGENT_LEVEL = _SUBAGENT_LEVELS[0]
# The rung's former name, still read: it is in the preferences file of everyone who
# set it, and an unknown rung is read as the weakest — which would take away a right
# the user had granted (client config.settings.SUBAGENT_LEVEL_ALIASES).
_SUBAGENT_LEVEL_ALIASES = {"parallel": "build"}
_SUBAGENT_LEVEL_META = "mimir/subagent_level"


def _probe_served_models() -> list[dict]:
    """The models the endpoint serves, asked once at start-up; [] when it lists none.

    Once, so the ``model`` description below is the same on every turn and the
    prompt prefix stays cacheable; a model added to the endpoint mid-session is
    offered from the next session on.
    """
    try:
        from mimir.client.llm.factory import get_backend
        return get_backend().served_model_info()
    except Exception as exc:  # a dead endpoint must not keep the server from starting
        print(f"spawn_agent: could not list served models: {exc}", file=sys.stderr)
        return []


def _model_description(served: list[dict]) -> str:
    try:
        from .model_catalog import describe_served_models
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
    """The budget as a person would say it. "about 120 minutes" is a number to convert
    before it means anything, and a working child's budget is now measured in hours."""
    if secs >= 5400:
        hours = secs / 3600
        return f"about {hours:.0f} hours" if hours >= 1.75 else "about an hour and a half"
    minutes = secs // 60
    return f"about {minutes} minutes" if minutes >= 2 else f"{secs} seconds"


def _prune_tools(agent, granted: list[str]) -> None:
    """Keep only *granted* on this child, whatever its servers advertised."""
    keep = set(granted)
    agent.tools = [t for t in agent.tools if t.get("function", {}).get("name") in keep]
    agent.tool_owner = {k: v for k, v in agent.tool_owner.items() if k in keep}
    agent.tool_caps = {k: v for k, v in agent.tool_caps.items() if k in keep}


def _write_record(session: str, record: dict) -> None:
    """Write this child's ``subagent.json`` card (task, grant, outcome) for the panel; best-effort."""
    if not session:
        return
    try:
        from .._shared.state_paths import session_dir
        write_json(os.path.join(session_dir(session), "subagent.json"), record,
                   ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"spawn_agent: could not record sub-session {session}: {exc}", file=sys.stderr)


def _repo_relative(paths: list[str], worktree: dict | None) -> list[str]:
    """*paths* inside the child's worktree, as the repository names them; others unchanged."""
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
    """Ask a child that overran its budget to stop (its ``cancel_flag``) and mark it abandoned.

    Its thread cannot be cancelled from here; the flag stops it at its next read.
    """
    child["abandoned"] = True
    agent = child.get("agent")
    flag = agent.turn.cancel_flag if agent is not None else None
    if flag is None:
        return
    try:
        flag.set()
    except Exception as exc:  # a child mid-teardown must not fail its caller
        print(f"spawn_agent: could not stop abandoned sub-agent: {exc}",
              file=sys.stderr)


def _overrun_note(budget: int) -> str:
    """Why this run ended, said once and used by both paths.

    A run stopped by its clock is not a run that finished, and a caller told only
    "here is what it did" reads the handoff as a conclusion. The distinction has to be
    in the words, because ``completed: False`` alone is a flag the model may not weigh.
    """
    return (
        f"This sub-agent was stopped after {_human_budget(budget)} — its time budget, "
        f"not the end of its work. What follows is where it had got to, not a "
        f"conclusion. Carry it on from here (its session holds its todo list and its "
        f"scratchpad) rather than starting the task again: a full budget has just been "
        f"spent on it and nothing about it has changed."
    )


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
    """What can be read off a child still running past its budget: where it got to."""
    agent = child.get("agent")
    worktree = child.get("worktree")
    files_read = _repo_relative(agent.carry.files_read() if agent else [], worktree)
    files_written = _repo_relative(agent.carry.files_written() if agent else [], worktree)
    lines = []
    if files_written:
        lines.append("Files it had already modified: " + ", ".join(files_written) + ".")
    if files_read:
        lines.append(f"It had read {len(files_read)} file(s).")
    lines.append("Its todo list and scratchpad are under its session, as it left them.")
    # Runs it launched and is no longer there to answer for. Read defensively, as the
    # agent above is: this is called from the caller's thread, and a child that never
    # got as far as a watcher has none to read. The plural key is what tells the client
    # these are runs to adopt rather than a run this call launched, and each descriptor
    # already names the directory the parent must read it through.
    watcher = child.get("watcher")
    handed_over: list[dict] = []
    try:
        handed_over = watcher.handover() if watcher is not None else []
    except Exception as exc:  # a child mid-teardown must not fail its caller
        print(f"spawn_agent: could not read a sub-agent's runs: {exc}", file=sys.stderr)
    if handed_over:
        keys = ", ".join(str(d.get("job_key") or "?") for d in handed_over)
        lines.append(f"Runs it left going, now watched here: {keys}.")
    out: dict[str, Any] = {
        "answer": " ".join(lines),
        "files_read": files_read,
        "files_written": files_written,
    }
    if handed_over:
        out["background_jobs"] = handed_over
    return out


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
    "more. Take as many tool-call steps as the work needs; what bounds you is the "
    "clock, and you have {budget}. Well before it runs out, STOP and finish with a "
    "HANDOFF — what you established and how you verified it, what is left, and where it "
    "is (files, branch, scratchpad). A caller who gets your handoff can carry the work "
    "on; a caller who gets nothing pays for it twice."
)

# ── Sub-agent output routing ──────────────────────────────────────────────────
# This server is a subprocess whose stdout IS the JSON-RPC pipe, so anything
# printed here corrupts the protocol. Two leaks are plugged: the child agent gets
# an event sink (without one, event_sink.emit falls back to printing every event),
# and stdout is pointed at stderr for whatever still prints.


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
    from .model_catalog import subagent_model_refusal
    return subagent_model_refusal(model, _SERVED_MODELS)


def _meta_value(ctx: Context | None, key: str):
    """One key of this call's request _meta, or None."""
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        return None
    return (getattr(meta, "model_extra", None) or {}).get(key)


def _caller_grantable(ctx: Context | None) -> dict[str, str]:
    """``{tool: server}`` the caller may grant, from the call's metadata; missing means nothing."""
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
    """How far the user lets this child go; the weakest rung when unsaid or unknown.

    Another name of a rung ("parallel") resolves here rather than being read as unknown,
    which would take away the right its user granted.
    """
    level = str(_meta_value(ctx, _SUBAGENT_LEVEL_META) or "").strip().lower()
    level = _SUBAGENT_LEVEL_ALIASES.get(level, level)
    return level if level in _SUBAGENT_LEVELS else _DEFAULT_SUBAGENT_LEVEL


def _level_allows(level: str, required: str) -> bool:
    level = _SUBAGENT_LEVEL_ALIASES.get(level, level)
    return _SUBAGENT_LEVELS.index(level) >= _SUBAGENT_LEVELS.index(required)


def _sub_session_id() -> str:
    """A session of this child's own: ``<parent>/subagents/sub-xxxxxxxx`` (directories only)."""
    child = f"sub-{uuid.uuid4().hex[:8]}"
    try:
        from .._shared.state_paths import active_session_id
        parent = active_session_id()
    except Exception:
        parent = ""
    return f"{parent}/subagents/{child}" if parent else child


# ── A copy of the code, for an axis of its own ────────────────────────────────
# A sub-agent asked for one gets a git worktree: the same repository, its own branch
# and its own directory, so it can edit, build and measure without the run next door
# reading its files. Only worth it when several axes run at once — hence a rung the
# user sets, and an argument the caller passes.


# ── Detached sub-agents ───────────────────────────────────────────────────────
# A background child outlives its call: the call returns a ``background_job`` descriptor
# the client's watcher polls, like a detached shell command. This registry is the
# process's memory of them, shared by the tool's thread and the loop (under the lock).


# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(kind="agent",
    caps=[DELEGATE, PLAN_READONLY, BACKGROUNDABLE],
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
    timeout_secs=spawn_limits.SUBAGENT_HARD_CAP_SECS + 60,
    label="Sub-agent: {task}",
    read_only=False,
))
async def spawn_agent(
    task: str,
    context: str = "",
    tools: Annotated[list[str], Field(description=_TOOLS_DESCRIPTION)] = [],
    max_steps: int = spawn_limits.SUBAGENT_MAX_STEPS,
    time_budget_secs: int = 0,
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
        max_steps:  A ceiling on tool-call steps. 0 (the default) is no ceiling —
                    what bounds a sub-agent is its clock, not a step count, and a
                    count cut long work off for a reason unrelated to its difficulty.
        time_budget_secs:
                    Wall time for this run. 0 (the default) takes the whole budget its
                    rung allows: 10 minutes for reconnaissance, 2 hours for a
                    sub-agent given a writing or executing tool — which is work that
                    builds and measures, and ten minutes of it is a run cut short for
                    no reason to do with the task. Pass a smaller number to bound a
                    run you expect to be short; a larger one is clamped. The sub-agent
                    is told its budget and is asked to hand over before it runs out.
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
                                  changed and a ``diffstat``. The copy is kept either
                                  way — it holds the build tree the axis paid for — and
                                  the oldest go when a later sub-agent needs the room;
                                  what tells a finished axis from an interrupted one is
                                  ``committed`` and the ``note``, not ``kept``. Read the
                                  branch, compare the axes, and merge the one you keep
                                  yourself.
        On failure (the sub-agent crashed or ran out of time):
            {"status": "error", "error": "…", "answer": "<partial>", ...}
            — distinct ``status`` so the orchestrator can branch on failure without
              string-matching the answer text.
    """
    grantable = _caller_grantable(ctx)
    requested = [str(t).strip() for t in (tools or []) if str(t).strip()]
    refusal = _grant_refusal(requested, grantable)
    if refusal:
        return err(refusal, **_NO_ANSWER)
    level = _subagent_level(ctx)
    working, budget = _budget(ctx, level, requested, time_budget_secs)
    model, refusal = _resolve_model(ctx, model)
    if refusal:
        return err(refusal, **_NO_ANSWER)

    spawn_output.silence_stdout_once()
    session = _sub_session_id()
    # Longer than a blocking call can hold (the client caps any declared timeout): detached
    # whether or not it was asked for, since the model cannot see that ceiling.
    detached_for_length = budget > spawn_limits.SUBAGENT_HARD_CAP_SECS
    run = _ChildRun(session=session, model=model, requested=requested, budget=budget,
                    background=background or detached_for_length)

    # A slot before anything is created or any clock starts: the wait is not the child's
    # time, and a worktree cut for a child that never runs is litter.
    if not await spawn_jobs.wait_for_slot():
        return err(
            f"no room to start another sub-agent: {spawn_jobs.SUBAGENT_MAX_CONCURRENT} are already "
            f"running ({spawn_jobs.busy_with()}). Do something else and delegate this when one "
            f"lands — you are resumed as each finishes — or do it yourself. Do not "
            f"retry this call immediately: nothing will have changed.",
            **_NO_ANSWER,
        )
    run.slot.take(task)

    # A child granted anything writing or executing works in its own copy of the
    # repository, so concurrent children cannot overwrite each other.
    if working:
        run.worktree, refusal = spawn_worktree.create_worktree(run.child_key, session)
        if run.worktree is None:
            run.slot.release()
            return err(refusal, **_NO_ANSWER)

    run.start(
        task=task, context=context, grantable=grantable, max_steps=max_steps,
        approval_mode=_caller_approval_mode(ctx), level=level,
        # The long-lived MCP session (one per stdio connection), read on this loop:
        # without one, the child runs unattended.
        ask_session=getattr(ctx, "session", None),
    )
    if run.job_key:
        return run.detached_reply(detached_for_length)
    if not await run.wait(ctx):
        return run.overrun_reply()
    return run.result_reply()


_NO_ANSWER: dict = {"answer": "", "completed": False, "files_read": [], "files_written": []}


def _grant_refusal(requested: list[str], grantable: dict[str, str]) -> str | None:
    unavailable = [t for t in requested if t not in grantable]
    if not unavailable:
        return None
    offer = ", ".join(sorted(grantable)) if grantable else "(none)"
    return (f"cannot grant {', '.join(unavailable)}: not available to you right now, or "
            f"reserved to you (planning, delegation, asking the user, writing memory, "
            f"cluster submission). What you may grant: {offer}. Leave `tools` empty for a "
            f"read-only exploration.")


def _budget(ctx: Context | None, level: str, requested: list[str],
            time_budget_secs: int) -> tuple[bool, int]:
    """Whether the child is working, and its wall time.

    A child with a writing or executing tool is working (it also gets a worktree);
    anything else is reconnaissance. Decided before the child connects any server.
    """
    working = _level_allows(level, "build") and bool(set(requested) & _caller_writers(ctx))
    ceiling = (spawn_limits.SUBAGENT_WORKING_BUDGET_SECS if working
               else spawn_limits.SUBAGENT_EXPLORE_BUDGET_SECS)
    budget = max(spawn_limits.SUBAGENT_MIN_BUDGET_SECS,
                 min(int(time_budget_secs or ceiling), ceiling))
    return working, budget


def _resolve_model(ctx: Context | None, model: str) -> tuple[str, str | None]:
    """The model the child runs on (the caller's unless another is named), or a refusal."""
    caller_model = _caller_model(ctx)
    model = (model or "").strip()
    if model and model != caller_model:
        return model, _refuse_model(model)
    return caller_model, None


class _SlotLease:
    """One of the server's sub-agent slots, given back exactly once."""

    def __init__(self, session: str) -> None:
        self.session = session
        self.held = False

    def take(self, task: str) -> None:
        self.held = True
        spawn_jobs.take_slot(self.session, task)

    def release(self) -> None:
        if self.held:
            self.held = False
            spawn_jobs.free_slot(self.session)


class _ChildRun:
    """One sub-agent run: its thread, its output queue, and the reply to the caller."""

    def __init__(self, *, session: str, model: str, requested: list[str], budget: int,
                 background: bool) -> None:
        self.session, self.model, self.requested, self.budget = (
            session, model, requested, budget)
        self.child_key = session.rsplit("/", 1)[-1]
        self.job_key = self.child_key if background else ""
        self.slot = _SlotLease(session)
        self.worktree: dict | None = None
        self.events: queue.Queue = queue.Queue(maxsize=spawn_output.CHILD_QUEUE_MAX)
        self.counters: dict = {"dropped": 0}
        self.state: dict = {"sent": 0, "dropped": 0}
        self.future: Future[dict] = Future()
        # The running child, published by its thread: what an overrun reports. Read
        # defensively, another thread writes it.
        self.child: dict = {}

    def start(self, *, task: str, context: str, grantable: dict[str, str],
              max_steps: int, approval_mode: str, level: str, ask_session: Any) -> None:
        """Run the child in its own thread and event loop.

        Its MCP exit stack must open and close in one task on one loop, and the hard cap
        must not become a cancellation inside a stdio_client. The thread releases the slot,
        so a detached child holds it for its whole life.
        """
        loop = asyncio.get_running_loop()

        def _thread_main() -> None:
            try:
                result = asyncio.run(_run_sub_agent(
                    task, context, self.requested, grantable, max_steps, approval_mode,
                    on_event=spawn_output.make_child_sink(
                        self.events, self.counters, self.job_key, self.session),
                    model=self.model, session=self.session, budget=self.budget,
                    child=self.child, level=level, worktree=self.worktree,
                    # Detached or not, the child can reach the person: the session
                    # outlives the request.
                    ask_session=ask_session, ask_loop=loop,
                ))
                self.future.set_result(result)
                if self.job_key:
                    spawn_jobs.settle_job(
                        self.job_key, "done" if not result.get("error") else "crashed", result)
            except Exception as exc:
                self.future.set_exception(exc)
                if self.job_key:
                    spawn_jobs.settle_job(self.job_key, "crashed",
                                          {"answer": "", "completed": False, "error": str(exc)})
            finally:
                self.slot.release()

        # Until the thread runs, nothing else gives the slot back.
        try:
            if self.job_key:
                # Before the thread starts: a watcher polling a key this process does not
                # know yet would read it as a job that never existed.
                spawn_jobs.register_job(self.job_key, {
                    "state": "running", "started": time.time(), "budget": self.budget,
                    "task": task, "session": self.session, "model": self.model,
                    "tools": self.requested, "phase": "", "result": None, "child": self.child,
                })
            threading.Thread(target=_thread_main, daemon=True,
                             name=spawn_jobs.THREAD_NAME_PREFIX + self.child_key).start()
        except BaseException:
            self.slot.release()
            raise

    def _identity(self) -> dict:
        return {"model": self.model, "tools": self.requested, "session": self.session}

    def detached_reply(self, detached_for_length: bool) -> dict:
        return ok({
            "answer": "", "completed": False,
            "files_read": [], "files_written": [], "blocked_by_mode": [],
            **self._identity(),
            "background_job": spawn_jobs.job_descriptor(self.job_key),
            "workspace": {"branch": self.worktree["branch"], "path": self.worktree["path"]}
                         if self.worktree else {},
            "note": (
                ("Detached automatically: its budget is longer than a blocking call "
                 "can hold, so waiting on it here was never possible. "
                 if detached_for_length else "")
                + "Started in the background; this call returns before the sub-agent "
                  "finishes. You are resumed with its answer when it does."
            ),
        })

    async def wait(self, ctx: Context | None) -> bool:
        """Forward the child's events until it ends or its budget (plus time spent on the
        user) runs out. True when it ended."""
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        self.state["last_activity"] = started
        deadline = loop.time() + self.budget
        while (not self.future.done()
               and loop.time() < deadline + spawn_user_channel.paused_secs(self.session)):
            await spawn_output.forward_pending(ctx, self.events, self.state)
            await spawn_output.maybe_heartbeat(ctx, self.state, started)
            await asyncio.sleep(spawn_output.POLL_SECS)
        # A last drain, on overrun too: the final tool result says where the run stopped.
        await spawn_output.forward_pending(ctx, self.events, self.state)
        dropped = self.counters.get("dropped", 0) + self.state.get("dropped", 0)
        if dropped:
            await spawn_output.report(ctx, self.state, {"v": 1, "t": "end", "dropped": dropped})
        return self.future.done()

    def overrun_reply(self) -> dict:
        """Stop the child, then hand back where it got to rather than a bare failure."""
        _abandon_child(self.child)
        partial = _partial_handoff(self.child)
        return err(
            _overrun_note(self.budget) + " Either carry it on yourself, or delegate one "
            "narrower step of it, saying in the task what this run established and "
            "which approaches not to retry.",
            answer=partial.pop("answer", ""), completed=False, **self._identity(),
            workspace=self.child.get("workspace", {}), **partial,
        )

    def result_reply(self) -> dict:
        try:
            result = self.future.result(timeout=0)
        except Exception as exc:
            return err(f"sub-agent failed: {exc}", **_NO_ANSWER, **self._identity(),
                       workspace=self.child.get("workspace", {}))
        files = {"files_read": result.get("files_read", []),
                 "files_written": result.get("files_written", [])}
        if result.get("error"):
            # The child's own run raised: an error, with the partial answer kept.
            return err(f"sub-agent crashed: {result['error']}",
                       answer=result.get("answer", ""), completed=False, **files,
                       blocked_by_mode=result.get("blocked_by_mode", []),
                       **self._identity(), workspace=result.get("workspace", {}))
        if not str(result.get("answer") or "").strip():
            # The answer is the payload: a blank one is not a successful delegation.
            return err("sub-agent returned no answer: nothing was delegated back. Re-issue "
                       "the call with a narrower, self-contained question, or do the work "
                       "yourself.", answer="", completed=False, **files, **self._identity())
        return ok({
            "answer": result["answer"],
            "completed": result["completed"],
            "files_read": result["files_read"],
            "files_written": result["files_written"],
            "blocked_by_mode": result["blocked_by_mode"],
            **self._identity(),
            "workspace": result.get("workspace", {}),
            # Runs it left going, under the plural key: the client adopts them for the
            # caller. Absent when there are none.
            **({"background_jobs": result["background_jobs"]}
               if result.get("background_jobs") else {}),
        })


@mcp.tool(**tool_caps(kind="agent",
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

    A handle this server no longer holds in memory — it was restarted since the run was
    detached — is answered from the card the sub-agent left, which carries its whole
    answer. ``outcome`` then says how the card itself put it, next to the state above.

    You are resumed automatically when a background sub-agent finishes, so do not poll
    its status: ask for its ``result`` if you need what it has so far, and otherwise
    get on with other work.

    Args:
        job_key: The handle spawn_agent(background=True) returned.
    """
    op = (op or "status").strip().lower()
    if op == "list":
        with spawn_jobs.JOBS_LOCK:
            jobs = [
                {"job_key": key, "state": spawn_jobs.job_state(entry), "task": entry.get("task", ""),
                 "session": entry.get("session", ""),
                 "elapsed_secs": int(time.time() - entry.get("started", 0))}
                for key, entry in spawn_jobs.JOBS.items()
            ]
        jobs.sort(key=lambda j: j["elapsed_secs"])
        return ok({"jobs": jobs, "count": len(jobs)})

    entry = spawn_jobs.job_snapshot(job_key) or spawn_jobs.entry_from_card(job_key)
    if entry is None:
        return err(f"unknown sub-agent job {job_key!r}: no sub-agent of this session "
                   f"was started under that handle, and none left a card under it. Use "
                   f"op='list' to see the ones this server still holds.")
    state = spawn_jobs.job_state(entry)
    if state == "unknown" and not entry.get("from_card"):
        # Past its budget and still going in its own thread. Nothing else notices:
        # the blocking path stops a child on its deadline, this one had no deadline to
        # sit on, so a detached run that overran carried on spending model calls and
        # tool budget on an answer nobody would ever read. Stopped here, at the first
        # read that sees the wall crossed — the watcher polls this, so that is soon.
        _abandon_child(entry.get("child") or {})
    base = {
        "state": state,
        "job_key": job_key,
        "task": entry.get("task", ""),
        "session": entry.get("session", ""),
        "model": entry.get("model", ""),
        "tools": entry.get("tools", []),
        # 0 when the start time is unknown — a card whose stamp did not parse would
        # otherwise report the seconds since 1970 as this run's duration.
        "elapsed_secs": int((entry.get("ended") or time.time()) - entry["started"])
                        if entry.get("started") else 0,
        # What it is doing, in the words of its own last tool row. The watcher passes
        # this on as the job's phase, which is the only sign of life a detached child
        # has once its call has returned.
        "phase": entry.get("phase", ""),
    }
    if entry.get("from_card"):
        # Rebuilt from the card: say so, and keep the card's own word for how it ended.
        # "abandoned" has no place in the watcher's vocabulary but every place in a
        # report, and a caller told only "crashed" would look for a crash there was not.
        base["outcome"] = entry.get("outcome", "")
        base["from_card"] = True
    if op != "result":
        return ok(base)
    result = entry.get("result")
    if result is None:
        # Still working, or stopped on its wall a moment ago: what it has touched is
        # real, its answer is not. A run the clock ended says so in as many words —
        # `completed: False` alone is a flag a reader can skim past, and a handoff read
        # as a conclusion is the whole failure this guards against.
        partial = _partial_handoff(entry.get("child") or {})
        if state == "unknown":
            budget = int(entry.get("budget") or spawn_limits.SUBAGENT_HARD_CAP_SECS)
            partial["answer"] = (_overrun_note(budget) + " " + partial["answer"]).strip()
            partial["ended_on_budget"] = True
        return ok({**base, **partial, "completed": False})
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
    budget: int = spawn_limits.SUBAGENT_EXPLORE_BUDGET_SECS,
    child: dict | None = None,
    level: str = _DEFAULT_SUBAGENT_LEVEL,
    worktree: dict | None = None,
    ask_session=None,
    ask_loop=None,
) -> dict:
    """Async implementation: create MimirAgent, wire tools, run query.

    Returns ``{"answer", "completed", "files_read", "files_written", "error"}``;
    ``error`` is None on a clean run (the task may still be incomplete — see
    ``completed``).
    """
    from mimir.client.agent_core import MimirAgent

    # Its own session (todo, scratchpad, approved paths), passed to the constructor: the
    # approval manager captures it there.
    agent = MimirAgent(model=model or _caller_model(None), session_id=session or None)
    # Read by every server it is about to start (server_manager merges this into their
    # environment), never by the servers already running for the caller.
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
    # Its own watcher, so a detached run it launches is held by something: without one
    # every backgroundable launch waits the run out in-turn, inside the child's budget,
    # and a run that outlives the budget outlives the only thing watching it. The child
    # hears about it through its steer channel, at its next step boundary.
    watcher = child_jobs.ChildWatcher(
        agent=agent, origin=spawn_user_channel.child_origin(agent, task), on_activity=on_event)
    agent.register_background_job = watcher.register
    agent.watched_background_jobs = watcher.live_descriptors
    agent.poll_steer = watcher.drain_steers
    if child is not None:
        child["agent"] = agent
        # Published for the same reason the agent is: a caller out of patience reads the
        # runs still in flight off this, from its own thread, and hands them over.
        child["watcher"] = watcher
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
            ask_session=ask_session, ask_loop=ask_loop)
        # A run it launched can outlive the answer it gives. Named on the result the
        # caller reads, because a run named nowhere in it is a run nobody is left
        # holding — and the plural key is what has the client adopt it for the parent
        # rather than read it as something this call started.
        handed_over = watcher.handover()
        if handed_over:
            result["background_jobs"] = handed_over
        return result
    finally:
        if worktree is not None:
            # Committed on a clean run, left staged when it did not finish: an
            # unfinished axis has its state nowhere else. The copy itself stays either
            # way — see finish_worktree.
            clean = bool(result and not result.get("error"))
            record["workspace"] = spawn_worktree.finish_worktree(worktree, task, clean)
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
            # cuts it for its own payload (session.store.list_subagents); cutting
            # here would cut it everywhere, for good.
            "answer": (result or {}).get("answer") or "",
            "files_written": (result or {}).get("files_written", []),
        })
        # Close the child's MCP sessions here, in the task that opened them (anyio's cancel
        # scope), after stopping its watchers, which poll through those servers.
        try:
            await watcher.close()
        except Exception as exc:  # a dropped watch must not mask the child's answer
            print(f"spawn_agent: sub-agent watcher warning: {exc}", file=sys.stderr)
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
    budget: int = spawn_limits.SUBAGENT_EXPLORE_BUDGET_SECS,
    level: str = _DEFAULT_SUBAGENT_LEVEL,
    worktree: dict | None = None,
    ask_session=None,
    ask_loop=None,
) -> dict:
    """Configure the child agent, run it, and report what it did."""
    from mimir.client.extensions import all_servers
    from mimir.client.tools.capabilities import (
        PLAN_BLOCKED,
        PLAN_READONLY,
        explorer_servers,
        has_cap,
    )

    # Sub-agents don't reason out loud: set the rung, not just the run() flag, since
    # the loop re-reads the agent's depth every step (the user steering the parent's
    # thinking must not leak into a child run).
    agent.apply_setting("thinking_depth", 0)
    # A conclusion is a paragraph, and a tool-call step is shorter still. Uncapped,
    # one step may claim the whole answer reserve — measured at ~40k tokens, i.e.
    # several minutes of generation during which the child emits nothing at all.
    agent.max_answer_tokens = spawn_limits.SUBAGENT_ANSWER_TOKENS
    # The user's approval mode, not the default: the child acts for the same user. It
    # has no one to ask (its stdin is this server's JSON-RPC pipe), so what the mode
    # does not cover is refused without a prompt and reported back as blocked.
    agent.apply_setting("approval_mode", approval_mode)
    agent.approvals.unattended = True
    if ask_session is not None and ask_loop is not None:
        # There is a person at the end of this call: its approvals and its questions go
        # to them, queued behind any other child's and saying which child is asking.
        spawn_user_channel.install_user_channel(agent, ask_session, ask_loop, task)
    # The child writes the shared path allowlist too; starting from what is already
    # there keeps its writes from erasing the caller's grants.
    agent.approvals.paths.allowed.update(approved_roots())

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

    def _unconnected(name: str, exc: BaseException) -> None:
        print(f"spawn_agent: could not connect server '{name}': {exc}", file=sys.stderr)

    await agent.connect_servers(servers_to_connect, on_error=_unconnected)

    if requested:
        _prune_tools(agent, requested)

    # Seed classification from this sub-agent's own (subset) registry — keeps
    # approval/plan-block/caching correct and per-agent (the sub-agent connects
    # fewer servers than the parent, and keeps fewer tools still).
    agent.seed_classification_from_caps()


    # What the child was given decides how it runs. Nothing that writes or executes
    # among its tools means nothing to gate: it explores, in a read-only mode. One such
    # tool and it works, which is the only way a granted shell can actually run a build.
    # Set on the agent as well as passed to run(), so the loop's mode tracking starts
    # where it ends and reads no switch on the first step.
    working = _level_allows(level, "build") and any(
        has_cap(name, PLAN_BLOCKED, agent.tool_caps)
        or has_cap(name, PLAN_READONLY, agent.tool_caps)
        for name in requested
    )
    mode = "agent" if working else _READONLY_CHILD_MODE
    agent.apply_setting("mode", mode)
    # Set here rather than with the rest of the child's setup: how much window the
    # child may fill follows from what it was given, and that is not known until its
    # tools have been classified just above.
    agent.max_context_tokens = (
        spawn_limits.SUBAGENT_CONTEXT_TOKENS_WORKING if working else spawn_limits.SUBAGENT_CONTEXT_TOKENS_EXPLORE
    )

    brief = (
        _WORK_BRIEF.format(budget=_human_budget(budget))
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

    # What the child touched this run. `files_read` goes back so the caller can record
    # the evidence a delegated sweep produced.
    files_read = _repo_relative(agent.carry.files_read(), worktree)
    files_written = _repo_relative(agent.carry.files_written(), worktree)
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
