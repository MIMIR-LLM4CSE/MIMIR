"""One sub-agent run as its caller holds it: the thread and loop it runs in, the wait, the
budget guard, and the reply — a result, a handoff on overrun, or a handle once detached."""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any

from mcp.server.fastmcp import Context

from ....shared.responses import err, ok
from . import build, human, jobs, limits, output
from .outcome import NO_ANSWER, abandon_child, human_budget, overrun_note, partial_handoff

# How often a run checks its budget, at most.
_BUDGET_CHECK_SECS = 30.0

class _SlotLease:
    """One of the server's sub-agent slots, given back exactly once."""

    def __init__(self, session: str) -> None:
        self.session = session
        self.held = False

    def take(self, task: str) -> None:
        self.held = True
        jobs.take_slot(self.session, task)

    def release(self) -> None:
        if self.held:
            self.held = False
            jobs.free_slot(self.session)


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
        self.events: queue.Queue = queue.Queue(maxsize=output.CHILD_QUEUE_MAX)
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
                result = asyncio.run(build.run_sub_agent(
                    task, context, self.requested, grantable, max_steps, approval_mode,
                    on_event=output.make_child_sink(
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
                    jobs.settle_job(self.job_key, "unknown", None)
                else:
                    jobs.settle_job(
                        self.job_key, "done" if not result.get("error") else "crashed", result)
            except Exception as exc:
                self.future.set_exception(exc)
                jobs.settle_job(self.job_key, "crashed",
                                      {"answer": "", "completed": False, "error": str(exc)})
            finally:
                self.slot.release()

        # Until the thread runs, nothing else gives the slot back.
        try:
            # Before the thread starts: a watcher polling a key this process does not
            # know yet would read it as a job that never existed.
            self.started = time.time()
            jobs.register_job(self.job_key, {
                "state": "running", "started": self.started, "budget": self.budget,
                "task": task, "session": self.session, "model": self.model,
                "tools": self.requested, "phase": "", "result": None, "child": self.child,
            })
            threading.Thread(target=_thread_main, daemon=True,
                             name=jobs.THREAD_NAME_PREFIX + self.child_key).start()
        except BaseException:
            self.slot.release()
            raise
        # Held here: the loop keeps only a weak reference to a task.
        self._guard = loop.create_task(self._stop_at_budget())

    def working_secs(self) -> float:
        """How long the child has worked: time in front of the user is not its time."""
        return time.time() - self.started - human.paused_secs(self.session)

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
            "background_job": jobs.job_descriptor(self.job_key),
            "workspace": {"branch": self.worktree["branch"], "path": self.worktree["path"]}
                         if self.worktree else {},
            "note": (
                f"Still working after {human_budget(limits.SUBAGENT_BLOCKING_WAIT_SECS)}, "
                f"longer than a call can wait: it carries on in the background, within its "
                f"budget of {human_budget(self.budget)}. You are resumed with its answer when "
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
        window = min(self.budget, limits.SUBAGENT_BLOCKING_WAIT_SECS)
        deadline = loop.time() + window
        while (not self.future.done()
               and loop.time() < deadline + human.paused_secs(self.session)):
            await output.forward_pending(ctx, self.events, self.state)
            await output.maybe_heartbeat(ctx, self.state, started)
            await asyncio.sleep(output.POLL_SECS)
        # A last drain, on overrun too: the final tool result says where the run stopped.
        await output.forward_pending(ctx, self.events, self.state)
        dropped = self.counters.get("dropped", 0) + self.state.get("dropped", 0)
        if dropped:
            await output.report(ctx, self.state, {"v": 1, "t": "end", "dropped": dropped})
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
