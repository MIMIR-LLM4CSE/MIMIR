"""One sub-agent run: the child agent built, configured and driven in a thread of its own,
what it is told, and what its caller gets back — a result, a handoff on overrun, or a
handle when it was detached.
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
from typing import Any, Callable

from mcp.server.fastmcp import Context

from mimir.client.core.ports import AgentHost
from . import caller, child_jobs
from . import spawn_jobs, spawn_limits, spawn_output, spawn_user_channel, spawn_worktree
from ...shared.approved_roots import approved_roots
from ...shared.atomic import write_json
from ...shared.responses import err, ok


# The read-only mode a child without any writing tool runs in (one of the client's
# config.models.READONLY_MODES). "ask" rather than "plan": the child answers a
# question, it does not draft a plan.
_READONLY_CHILD_MODE = "ask"

# How often a run checks its budget, at most.
_BUDGET_CHECK_SECS = 30.0


def _human_budget(secs: int) -> str:
    """The budget as a person would say it: "about 120 minutes" is a number to convert
    before it means anything."""
    if secs >= 3600:
        hours = secs / 3600
        if hours < 1.25:
            return "about an hour"
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
        from ...shared.state_paths import session_dir
        write_json(os.path.join(session_dir(session), "subagent.json"), record,
                   ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"spawn_agent: could not record sub-session {session}: {exc}", file=sys.stderr)


def repo_relative(paths: list[str], worktree: dict | None) -> list[str]:
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


def abandon_child(child: dict) -> None:
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


def overrun_note(budget: int) -> str:
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


def final_state(child: dict | None, result: dict | None) -> str:
    """The terminal state to record for a finished child.

    "abandoned" outranks the other two: a child whose caller timed out may still
    reach a clean result seconds later, and writing that as "finished" tells the
    panel a run succeeded when its answer went nowhere.
    """
    if child is not None and child.get("abandoned"):
        return "abandoned"
    return "finished" if result else "failed"


def partial_handoff(child: dict) -> dict:
    """What can be read off a child still running past its budget: where it got to."""
    agent = child.get("agent")
    worktree = child.get("worktree")
    files_read = repo_relative(agent.carry.files_read() if agent else [], worktree)
    files_written = repo_relative(agent.carry.files_written() if agent else [], worktree)
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
WORK_BRIEF = (
    "You are a working sub-agent: one self-contained piece of a larger task, with your "
    "own context and your own todo list. Only the tools listed for you are available — "
    "nothing else is connected, so plan within them and say so if the task truly needs "
    "more. Take as many tool-call steps as the work needs; what bounds you is the "
    "clock, and you have {budget}. Well before it runs out, STOP and finish with a "
    "HANDOFF — what you established and how you verified it, what is left, and where it "
    "is (files, branch, scratchpad). A caller who gets your handoff can carry the work "
    "on; a caller who gets nothing pays for it twice."
)

def new_sub_session() -> str:
    """A session of this child's own: ``<parent>/subagents/sub-xxxxxxxx`` (directories only)."""
    child = f"sub-{uuid.uuid4().hex[:8]}"
    try:
        from ...shared.state_paths import active_session_id
        parent = active_session_id()
    except Exception:
        parent = ""
    return f"{parent}/subagents/{child}" if parent else child

NO_ANSWER: dict = {"answer": "", "completed": False, "files_read": [], "files_written": []}

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


class ChildRun:
    """One sub-agent run: its thread, its output queue, and the reply to the caller."""

    def __init__(self, *, session: str, model: str, requested: list[str], budget: int) -> None:
        self.session, self.model, self.requested, self.budget = (
            session, model, requested, budget)
        self.child_key = session.rsplit("/", 1)[-1]
        # Every run is a job: one the caller waits on can still carry on detached.
        self.job_key = self.child_key
        self.slot = _SlotLease(session)
        self.worktree: dict | None = None
        self.events: queue.Queue = queue.Queue(maxsize=spawn_output.CHILD_QUEUE_MAX)
        self.counters: dict = {"dropped": 0}
        self.state: dict = {"sent": 0, "dropped": 0}
        self.future: Future[dict] = Future()
        # The running child, published by its thread: what an overrun reports. Read
        # defensively, another thread writes it.
        self.child: dict = {}
        self.started = time.time()
        self._guard: asyncio.Task | None = None

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
                result = asyncio.run(run_sub_agent(
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
                if self.child.get("abandoned"):
                    # Stopped on its budget: what it got to is read off the child, with
                    # the note that says the clock ended it (subagent_job).
                    spawn_jobs.settle_job(self.job_key, "unknown", None)
                else:
                    spawn_jobs.settle_job(
                        self.job_key, "done" if not result.get("error") else "crashed", result)
            except Exception as exc:
                self.future.set_exception(exc)
                spawn_jobs.settle_job(self.job_key, "crashed",
                                      {"answer": "", "completed": False, "error": str(exc)})
            finally:
                self.slot.release()

        # Until the thread runs, nothing else gives the slot back.
        try:
            # Before the thread starts: a watcher polling a key this process does not
            # know yet would read it as a job that never existed.
            self.started = time.time()
            spawn_jobs.register_job(self.job_key, {
                "state": "running", "started": self.started, "budget": self.budget,
                "task": task, "session": self.session, "model": self.model,
                "tools": self.requested, "phase": "", "result": None, "child": self.child,
            })
            threading.Thread(target=_thread_main, daemon=True,
                             name=spawn_jobs.THREAD_NAME_PREFIX + self.child_key).start()
        except BaseException:
            self.slot.release()
            raise
        # Held here: the loop keeps only a weak reference to a task.
        self._guard = loop.create_task(self._stop_at_budget())

    def working_secs(self) -> float:
        """How long the child has worked: time in front of the user is not its time."""
        return time.time() - self.started - spawn_user_channel.paused_secs(self.session)

    async def _stop_at_budget(self) -> None:
        """The hard cap, whoever is waiting on the child or not: stopped at its budget."""
        while not self.future.done():
            left = self.budget - self.working_secs()
            if left <= 0:
                abandon_child(self.child)
                return
            await asyncio.sleep(min(left, _BUDGET_CHECK_SECS))

    def _identity(self) -> dict:
        return {"model": self.model, "tools": self.requested, "session": self.session}

    def detached_reply(self, still_working: bool = False) -> dict:
        return ok({
            "answer": "", "completed": False,
            "files_read": [], "files_written": [], "blocked_by_mode": [],
            **self._identity(),
            "background_job": spawn_jobs.job_descriptor(self.job_key),
            "workspace": {"branch": self.worktree["branch"], "path": self.worktree["path"]}
                         if self.worktree else {},
            "note": (
                f"Still working after {_human_budget(spawn_limits.SUBAGENT_BLOCKING_WAIT_SECS)}, "
                f"longer than a call can wait: it carries on in the background, within its "
                f"budget of {_human_budget(self.budget)}. You are resumed with its answer when "
                f"it finishes; work on something else meanwhile."
                if still_working else
                "Started in the background; this call returns before the sub-agent "
                "finishes. You are resumed with its answer when it does."
            ),
        })

    async def wait(self, ctx: Context | None) -> str:
        """Forward the child's events until it ends, its budget runs out or the call can wait
        no longer (time spent on the user not counted).

        ``"done"``, ``"out_of_budget"`` or ``"still_working"``.
        """
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        self.state["last_activity"] = started
        window = min(self.budget, spawn_limits.SUBAGENT_BLOCKING_WAIT_SECS)
        deadline = loop.time() + window
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
        if self.future.done():
            return "done"
        return "out_of_budget" if window == self.budget else "still_working"

    def overrun_reply(self) -> dict:
        """Stop the child, then hand back where it got to rather than a bare failure."""
        abandon_child(self.child)
        partial = partial_handoff(self.child)
        return err(
            overrun_note(self.budget) + " Either carry it on yourself, or delegate one "
            "narrower step of it, saying in the task what this run established and "
            "which approaches not to retry.",
            answer=partial.pop("answer", ""), completed=False, **self._identity(),
            workspace=self.child.get("workspace", {}), **partial,
        )

    def result_reply(self) -> dict:
        try:
            result = self.future.result(timeout=0)
        except Exception as exc:
            return err(f"sub-agent failed: {exc}", **NO_ANSWER, **self._identity(),
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
    budget: int = spawn_limits.SUBAGENT_EXPLORE_BUDGET_SECS,
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
    # Its own watcher, so a detached run it launches is held by something: without one
    # every backgroundable launch waits the run out in-turn, inside the child's budget,
    # and a run that outlives the budget outlives the only thing watching it. The child
    # hears about it through its steer channel, at its next step boundary.
    watcher = child_jobs.ChildWatcher(
        agent=agent, origin=spawn_user_channel.child_origin(agent, task), on_activity=on_event)
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
    budget: int = spawn_limits.SUBAGENT_EXPLORE_BUDGET_SECS,
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
        spawn_limits.SUBAGENT_CONTEXT_TOKENS_WORKING if working else spawn_limits.SUBAGENT_CONTEXT_TOKENS_EXPLORE
    )

    brief = (
        WORK_BRIEF.format(budget=_human_budget(budget))
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
