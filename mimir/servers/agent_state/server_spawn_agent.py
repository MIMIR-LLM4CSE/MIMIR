"""MCP Spawn-Agent Server: delegate a sub-task to a fresh MimirAgent and return its answer.

Async, so several calls in one step run concurrently; the child's tool calls stream back
as progress. It gets only the tools the caller grants (and their servers), on the
caller's model unless another served one is named. The sub-agent itself is the
``subagent`` package: its caller (``caller``), its run (``run``, ``build``), its copy of
the repository (``worktree``).
"""

from __future__ import annotations

import time
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from .subagent import caller, jobs, limits, output
from .model_catalog import probe_served_models, served_models_description
from .subagent.build import new_sub_session
from .subagent.outcome import NO_ANSWER, abandon_child, overrun_note, partial_handoff
from .subagent.run import ChildRun
from .subagent.worktree import create_worktree
from ...shared.capabilities import (
    BACKGROUNDABLE,
    DELEGATE,
    MAIN_ONLY,
    PLAN_READONLY,
    REVERSIBLE,
    tool_caps,
)
from ...shared.responses import err, ok

mcp = FastMCP("spawn_agent")

# Asked once at start-up, so the ``model`` description is the same on every turn and the
# prompt prefix stays cacheable; a model added to the endpoint mid-session is offered
# from the next session on.
_SERVED_MODELS: list[dict] = probe_served_models()
_MODEL_DESCRIPTION = served_models_description(_SERVED_MODELS)

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
    # Deliberately above SUBAGENT_BLOCKING_WAIT_SECS, so this server answers first (the
    # child's result, or its handle once it carries on detached) instead of the
    # dispatcher killing the call with nothing to show.
    timeout_secs=limits.SUBAGENT_BLOCKING_WAIT_SECS + 60,
    label="Sub-agent: {task}",
    read_only=False,
))
async def spawn_agent(
    task: str,
    context: str = "",
    tools: Annotated[list[str], Field(description=_TOOLS_DESCRIPTION)] = [],
    max_steps: int = limits.SUBAGENT_MAX_STEPS,
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
                    Wall time for this run, a hard cap: at it the sub-agent is stopped
                    and hands back where it got to. 0 (the default) takes the whole
                    budget its rung allows: 1 hour for reconnaissance, 8 hours for a
                    sub-agent given a writing or executing tool. Pass a smaller number
                    to bound a run you expect to be short; a larger one is clamped. The
                    sub-agent is told its budget and is asked to hand over before it
                    runs out. A call still working after about 19 minutes returns a
                    handle and the sub-agent carries on in the background; you are
                    resumed with its answer when it finishes.
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
    grantable = caller.grantable(ctx)
    requested = [str(t).strip() for t in (tools or []) if str(t).strip()]
    refusal = caller.grant_refusal(requested, grantable)
    if refusal:
        return err(refusal, **NO_ANSWER)
    level = caller.subagent_level(ctx)
    working, budget = caller.budget(ctx, level, requested, time_budget_secs)
    model, refusal = caller.resolve_model(ctx, model, _SERVED_MODELS)
    if refusal:
        return err(refusal, **NO_ANSWER)

    # This server's stdout IS the JSON-RPC pipe, so anything printed corrupts the
    # protocol: stdout goes to stderr, and the child gets an event sink (without one,
    # event_sink.emit falls back to printing every event).
    output.silence_stdout_once()
    session = new_sub_session()
    run = ChildRun(session=session, model=model, requested=requested, budget=budget)

    # A slot before anything is created or any clock starts: the wait is not the child's
    # time, and a worktree cut for a child that never runs is litter.
    if not await jobs.wait_for_slot():
        return err(
            f"no room to start another sub-agent: {jobs.SUBAGENT_MAX_CONCURRENT} are already "
            f"running ({jobs.busy_with()}). Do something else and delegate this when one "
            f"lands — you are resumed as each finishes — or do it yourself. Do not "
            f"retry this call immediately: nothing will have changed.",
            **NO_ANSWER,
        )
    run.slot.take(task)

    # A child granted anything writing or executing works in its own copy of the
    # repository, so concurrent children cannot overwrite each other.
    if working:
        run.worktree, refusal = create_worktree(run.child_key, session)
        if run.worktree is None:
            run.slot.release()
            return err(refusal, **NO_ANSWER)

    run.start(
        task=task, context=context, grantable=grantable, max_steps=max_steps,
        approval_mode=caller.approval_mode(ctx), level=level,
        # The long-lived MCP session (one per stdio connection), read on this loop:
        # without one, the child runs unattended.
        ask_session=getattr(ctx, "session", None),
    )
    if background:
        return run.detached_reply()
    ended = await run.wait(ctx)
    if ended == "done":
        return run.result_reply()
    if ended == "out_of_budget":
        return run.overrun_reply()
    # Longer than a call can wait (the client caps any declared timeout): it carries on
    # detached, and the caller is resumed when it lands.
    return run.detached_reply(still_working=True)


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
        with jobs.JOBS_LOCK:
            listed = [
                {"job_key": key, "state": jobs.job_state(entry), "task": entry.get("task", ""),
                 "session": entry.get("session", ""),
                 "elapsed_secs": int(time.time() - entry.get("started", 0))}
                for key, entry in jobs.JOBS.items()
            ]
        listed.sort(key=lambda j: j["elapsed_secs"])
        return ok({"jobs": listed, "count": len(listed)})

    entry = jobs.job_snapshot(job_key) or jobs.entry_from_card(job_key)
    if entry is None:
        return err(f"unknown sub-agent job {job_key!r}: no sub-agent of this session "
                   f"was started under that handle, and none left a card under it. Use "
                   f"op='list' to see the ones this server still holds.")
    state = jobs.job_state(entry)
    if state == "unknown" and not entry.get("from_card"):
        # Past its budget and still going in its own thread: its run's budget guard
        # stops it, and so does this read when it sees the wall first.
        abandon_child(entry.get("child") or {})
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
        partial = partial_handoff(entry.get("child") or {})
        if state == "unknown":
            budget = int(entry.get("budget") or limits.SUBAGENT_EXPLORE_BUDGET_SECS)
            partial["answer"] = (overrun_note(budget) + " " + partial["answer"]).strip()
            partial["ended_on_budget"] = True
        return ok({**base, **partial, "completed": False})
    return ok({**base, **result, "completed": bool(result.get("completed"))})


if __name__ == "__main__":
    mcp.run(transport="stdio")
