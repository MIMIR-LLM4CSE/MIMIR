"""One sub-agent run: the child agent built, configured and driven in a thread of its own,
what it is told, and what its caller gets back — a result, a handoff on overrun, or a
handle when it was detached.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Callable


from mimir.client.core.ports import AgentHost
from mimir.client.session.journal import TranscriptLog
from . import caller, human, limits
from .brief import COPY_BRIEF, EXPLORE_BRIEF, WORK_BRIEF
from .outcome import final_state, human_budget, repo_relative
from .watcher import ChildWatcher
from .worktree import finish_worktree
from ....shared.approved_roots import approved_roots
from ....shared.atomic import write_json


# The read-only mode a child without any writing tool runs in (one of the client's
# config.models.READONLY_MODES). "ask" rather than "plan": the child answers a
# question, it does not draft a plan.
_READONLY_CHILD_MODE = "ask"

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
        from ....shared.state_paths import session_dir
        write_json(os.path.join(session_dir(session), "subagent.json"), record,
                   ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"spawn_agent: could not record sub-session {session}: {exc}", file=sys.stderr)

def new_sub_session() -> str:
    """A session of this child's own: ``<parent>/subagents/sub-xxxxxxxx`` (directories only)."""
    child = f"sub-{uuid.uuid4().hex[:8]}"
    try:
        from ....shared.state_paths import active_session_id
        parent = active_session_id()
    except Exception:
        parent = ""
    return f"{parent}/subagents/{child}" if parent else child

def _journaled(sink: Callable[[dict], None] | None,
               journal: TranscriptLog) -> Callable[[dict], None]:
    """*sink*, with every event written to *journal* first (which never raises)."""
    def _tee(event: dict) -> None:
        journal.append(event)
        if sink is not None:
            sink(event)
    return _tee


async def run_sub_agent(
    task: str,
    context: str,
    requested: list[str],
    grantable: dict[str, str],
    max_steps: int,
    approval_mode: str = "manual",
    on_event: Callable[[dict], None] | None = None,
    model: str = "",
    session: str = "",
    budget: int = limits.SUBAGENT_EXPLORE_BUDGET_SECS,
    child: dict | None = None,
    level: str = caller.DEFAULT_SUBAGENT_LEVEL,
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
    agent = MimirAgent(model=model or caller.model_of(None), session_id=session or None)
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
    # Its journal, kept as a conversation's is (``session/journal``): the task, every
    # event of its run, its answer. What audits a conversation reads a child too.
    journal = TranscriptLog()
    journal.bind(session or None)
    journal.append({"type": "query", "text": task})
    on_event = _journaled(on_event, journal)
    # Its own watcher, so a detached run it launches is held by something: without one
    # every backgroundable launch waits the run out in-turn, inside the child's budget,
    # and a run that outlives the budget outlives the only thing watching it. The child
    # hears about it through its steer channel, at its next step boundary.
    watcher = ChildWatcher(
        agent=agent, origin=human.child_origin(agent, task), on_activity=on_event)
    agent.attach(AgentHost(human=agent.host.human, watcher=watcher,
                           steer=watcher.drain_steers))
    if child is not None:
        child["agent"] = agent
        # Published for the same reason the agent is: a caller out of patience reads the
        # runs still in flight off this, from its own thread, and hands them over.
        child["watcher"] = watcher
        # Also the copy it works in: a run that overruns its budget is read through
        # partial_handoff, and that is where the paths it reports get their names.
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
        result = await drive_sub_agent(
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
        journal.append({"type": "answer", "text": result.get("answer") or "",
                        "completed": bool(result.get("completed"))})
        return result
    finally:
        if worktree is not None:
            # Committed on a clean run, left staged when it did not finish: an
            # unfinished axis has its state nowhere else. The copy itself stays either
            # way — see finish_worktree.
            clean = bool(result and not result.get("error"))
            record["workspace"] = finish_worktree(worktree, task, clean)
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
            # neither: abandon_child marked it on the way out, and its answer — clean
            # or not — was never delivered, so it never counts as completed.
            "state": final_state(child, result),
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


async def drive_sub_agent(
    agent,
    task: str,
    context: str,
    requested: list[str],
    grantable: dict[str, str],
    max_steps: int,
    on_event: Callable[[dict], None] | None = None,
    approval_mode: str = "manual",
    budget: int = limits.SUBAGENT_EXPLORE_BUDGET_SECS,
    level: str = caller.DEFAULT_SUBAGENT_LEVEL,
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
    agent.max_answer_tokens = limits.SUBAGENT_ANSWER_TOKENS
    # The user's approval mode, not the default: the child acts for the same user. It
    # has no one to ask (its stdin is this server's JSON-RPC pipe), so what the mode
    # does not cover is refused without a prompt and reported back as blocked.
    agent.apply_setting("approval_mode", approval_mode)
    agent.approvals.unattended = True
    if ask_session is not None and ask_loop is not None:
        # There is a person at the end of this call: its approvals and its questions go
        # to them, queued behind any other child's and saying which child is asking.
        human.install_user_channel(agent, ask_session, ask_loop, task)
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
    working = caller.level_allows(level, "build") and any(
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
        limits.SUBAGENT_CONTEXT_TOKENS_WORKING if working else limits.SUBAGENT_CONTEXT_TOKENS_EXPLORE
    )

    brief = (
        WORK_BRIEF.format(budget=human_budget(budget))
        if working else EXPLORE_BRIEF
    )
    if worktree:
        brief += COPY_BRIEF.format(path=worktree["path"], branch=worktree["branch"])
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
    files_read = repo_relative(agent.carry.files_read(), worktree)
    files_written = repo_relative(agent.carry.files_written(), worktree)
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
