"""Leaving the server running without this window, taking it back, stopping it.

A part of ``_Session`` (``ws_session``): a mixin over the connection's own state, so a
handler reads and writes the session it serves. ``MESSAGES`` names the inbound message
types it answers.
"""
from __future__ import annotations

import json
import logging
import os

from .. import server_registry
from .._ws_runtime import (
    _MIMIR_DIR_WS,
)
from ..detach import detach as detach_process
from ..job_scan import establish_baseline, has_baseline, scan_all_sessions
from ..ws_pool import idle_stop_reason

logger = logging.getLogger(__name__)


class DetachHandlers:
    MESSAGES: dict[str, str] = {
        "detach": "_handle_detach",
        "shutdown": "_handle_shutdown",
    }

    async def _report_ended_jobs(self) -> None:
        """Announce every detached run that finished while nothing was watching it.

        This is what makes coming back enough. A background run survives anything — its
        own process session, a trap that records the exit code, a descriptor on disk —
        but the *watcher* that promised to report it is a task on a worker's loop, and a
        restart takes it with them. Re-making that promise already existed, through a
        status tool returning ``background_jobs``, but it needed a turn in which somebody
        asked; here nobody has to.

        Handed to the pool's wake router, deliberately: a run reported late must be
        indistinguishable from one reported on time, so it goes through the same routing,
        the same wake text and the same coalescing. A conversation with no agent yet keeps
        its wake pending rather than losing it, and its next turn carries it.

        Run as a task rather than awaited in the handshake: it walks every session's
        job directories, and nothing must be able to come between a client connecting
        and the loop that reads its messages.

        Marked on disk as it is announced, because two places look for these — a
        connection arriving and a worker being built — and a run announced twice is a
        conversation woken twice for one build.

        A session never scanned before gets a *baseline* instead of a backlog. A
        detached job's directory is never swept, however old, so a long-lived workspace
        holds every build it ever ran, and an unmarked one there is as likely to have
        been reported by its own watcher as to be owed a wake. Reading that history as
        wakes owed wakes a conversation for a two-month-old build — worse than missing a
        recent one, and unlike a missed wake unbounded. Per session, and a sub-agent's
        counts as its own: a child seen here for the first time brings no backlog
        either.
        """
        try:
            found = scan_all_sessions()
        except Exception:
            logger.warning("job scan: the sessions could not be read", exc_info=True)
            return
        for session_id, jobs in found.items():
            if not has_baseline(session_id):
                establish_baseline(session_id, jobs)
                continue
            for job in jobs:
                if job.live:
                    continue   # a worker being built is what puts a watcher back on it
                descriptor = job.descriptor()
                self.pool.wakes.consume({
                    "type": "job_complete",
                    "job_key": job.job_key,
                    "server": descriptor.get("server"),
                    "kind": descriptor.get("kind"),
                    "state": job.state,
                    # Who gets woken, which for a sub-agent's run is not the session
                    # holding the files: a sub-session is a directory and no front end
                    # can load one, so a child's run is announced in the conversation
                    # that delegated it. Its markers stay under the child, where the
                    # scan that found it will look again.
                    "session_id": job.owner_session,
                    "run_session": job.session_id,
                    "summary": {"command": job.command, "exit_code": job.exit_code},
                    "status_op": job.status_op(),
                })

    async def _restore_detached_autonomy(self) -> None:
        """Take a run that was left going back, in the mode it was left under.

        A window opening onto a run that never stopped is the user who left it coming
        back, so attaching *is* the take-back: the claim that kept the process alive is
        dropped here, and the panel is told the server is attached. That is what makes
        closing this window end the run again — a server no conversation claims stops
        once its last client has been gone for the grace, and a claim nothing drops on
        arrival makes the process immortal, since a window that merely attached to a
        server holds no handle on it to kill.

        The level is a separate question, and the run's own answer is the truth: it is
        read before the claim goes and applied to the agents. Two things disagree with
        it otherwise — a webview arrives on its default, and a worker rebuilt during the
        absence comes up on the pool-wide record rather than the level chosen for this
        session, which would quietly drop a run from ``auto`` to ``manual`` and park it
        at its next sensitive call, with nothing said.

        Scoped to the run, not persisted past it: the level is read from the registry
        entry, which is cleared on a clean stop and liveness-checked. That is what keeps
        this from becoming the thing the approval mode is deliberately never persisted
        for — a mode that suppresses prompts being inherited by a later session that
        never asked for it.
        """
        held = {sid: lvl for sid, lvl in server_registry.claims().items()
                if lvl in ("manual", "auto", "auto_all")}
        if not held:
            return
        # Each conversation gets back the level *it* was left under. Applied to its own
        # agent, and recorded pool-wide only for the conversation on screen, which is
        # the one a worker built next most likely belongs to; every other claimed
        # conversation has its agent here already, since a claim is what kept it.
        for sid, lvl in held.items():
            worker = self.pool.get(sid)
            if worker is None:
                continue
            try:
                worker.apply_setting("approval_mode", lvl)
            except Exception:
                logger.warning("reattach: session %s refused the level %s it was left "
                               "under", sid, lvl, exc_info=True)
        level = held.get(self._active_session_id or "") or sorted(held.values())[-1]
        self._apply_setting("approval_mode", level)
        # Before anything is announced, because what the panel is about to be told —
        # that this window owns the server — has to be true of the registry the idle
        # sweep reads, and not only of the message. Every claim at once: one process
        # serves the whole workspace and this window is the only one looking at it, so
        # there is no conversation left for a claim to be held on behalf of.
        try:
            server_registry.unclaim(None)
        except Exception:
            # Reported and stepped over: the level below is the half the user can be
            # harmed by losing, and a claim that could not be dropped costs a process
            # that outlives this window — recoverable with the stop command.
            logger.warning("reattach: the detach claims could not be dropped",
                           exc_info=True)
        try:
            # That the server is this window's again, not only the level the run is
            # under. The panel decides from this what disconnecting costs, and it now
            # costs the runs: a window told otherwise would offer to walk away from
            # turns that end the moment it does.
            await self.ws.send(json.dumps({
                "type": "detached", "detached": False, "log": None,
                "autonomy": level, "sessions": [], "pid": os.getpid(),
                "setsid": False,
            }))
            await self.ws.send(json.dumps({"type": "approval_mode", "mode": level}))
            # Rendered, not notified: ``_notify`` writes to the transcript's transient
            # channel, which is dropped. Both halves of this are the kind of thing a
            # user walks back in needing to know — that the run has been approving
            # sensitive calls on its own, and that closing this window now ends it —
            # and neither is chatter.
            #
            # Quiet, though, and that is the point. It is not a warning: nothing has
            # gone wrong, nothing needs answering, and the user cannot act on it except
            # through controls already on screen. A badge and a coloured border read as
            # a problem to deal with on a conversation the user has just walked back
            # into — the one moment they are reading for what happened rather than for
            # what to do. So it reads as a rule across the thread, marking where they
            # rejoined and under what.
            #
            # Transient: it describes the state at this attachment, not anything that
            # happened in the conversation. Stored, every later load would replay it —
            # and a conversation rejoined twenty times would open on twenty of them,
            # each claiming to be now.
            #
            # It opens with the state the toggle shows — "attached" — because that is
            # the word the button answers to, and a line that only says what the run
            # has been doing leaves the user to work out which of the two positions
            # they are in. Then the consequence, which the glyph cannot show: the
            # process survived their absence and will not survive the next one unless
            # they ask again.
            #
            # The level is named only when something was deciding on its own, and the
            # count only when more than one conversation was left running: the levels
            # can differ, and "under auto_all" says nothing about which of them.
            under = f", under “{level}”" if level != "manual" else ""
            title = (
                f"attached — this conversation kept working with no window open"
                f"{under}; closing this window now stops it"
                if len(held) == 1 else
                f"attached — {len(held)} conversations kept working with no window "
                f"open, this one{under or ' at manual'}; closing this window now "
                f"stops them")
            await self._command_reply("detach", title, tone="quiet", transient=True)
        except Exception:
            logger.debug("_restore_detached_autonomy: the client went away mid-send", exc_info=True)
            return
        logger.info("reattach: %d claim(s) taken back; autonomy restored to %s from "
                    "the detached run", len(held), level)

    async def _handle_shutdown(self, msg: dict) -> None:
        """Stop the server, now or once it has nothing left to do.

        ``force`` stops it whatever is in flight — the user's own decision about their
        own machine, and the only way to end a run that is waiting on an answer they
        have decided not to give. Without it the request is refused while anything is
        still working, and the reasons are sent back: an arrest that silently discarded
        a two-hour build would be the worst answer this path could give.

        Either way it goes through the same exit as a signal, which is the only thing
        that closes the agents' MCP servers.
        """
        report = self.pool.idle_report()
        force = bool(msg.get("force"))
        if not report["idle"] and not force:
            try:
                await self.ws.send(json.dumps({
                    "type": "shutdown_refused",
                    "reasons": report["reasons"],
                }))
            except Exception:
                logger.debug("_handle_shutdown: the client went away mid-send", exc_info=True)
            return
        try:
            await self.ws.send(json.dumps({
                "type": "shutting_down",
                "forced": force,
                "reasons": report["reasons"],
            }))
        except Exception:
            logger.debug("_handle_shutdown: the client went away mid-send", exc_info=True)
        logger.info("shutdown: requested by the client (forced=%s; %s)", force,
                    "; ".join(report["reasons"]) or "nothing was running")
        self.pool.request_stop()

    def _safe_list_sessions(self) -> list:
        """The stored conversations, or none when the directory cannot be read.

        Guarded because its one caller is deciding what to claim: a listing that
        raises must not turn a detachment into an error, and an unreadable store is
        the one case where "this names no conversation" cannot be concluded — so it
        falls back to the agents, which is the stricter answer.
        """
        try:
            return self.store.list_sessions()
        except Exception:
            logger.warning("detach: the stored conversations could not be listed",
                           exc_info=True)
            return []

    async def _handle_detach(self, msg: dict) -> None:
        """Let the server go on without this window — "continue without me".

        Served here, on the WS loop, because the two things that make this process
        survivable are things it does to *itself*: re-pointing its output fds away from
        the pipe the extension host holds, and leaving that host's process group. The
        spawn is not touched at all, which is why this works on a server that was
        started as an ordinary child — and why it can be decided when the user is
        leaving rather than when they connected, which is the moment they actually know
        whether anything is worth leaving running.

        The autonomy level is the user's answer to "what may it do while I am gone", and
        it is required rather than defaulted: a detachment at ``manual`` parks at the
        first sensitive tool and does almost nothing overnight, and that has to be a
        choice rather than something that happened to them. It is applied to the
        conversations named, through the same seam ``/approvals`` uses, so a turn
        already in flight picks the new level up at its next gate.

        Naming no conversation means all of them, and then the level is recorded
        pool-wide as well — which is the only form that survives a worker being rebuilt,
        since the pool's record of UI settings is per-pool and not per-session.
        """
        # Giving the decision back. The process-level work cannot be undone — fds that
        # point at a log file have no pipe to return to — but none of it needs undoing:
        # what makes a server survive a window closing is that nobody kills it, and that
        # is a decision, not a state of the process. So this clears the claim and leaves
        # the redirect in place.
        if msg.get("enabled") is False:
            # Per conversation, like the claim it undoes: naming some takes back only
            # those, naming none takes back every one. The process goes on surviving
            # for as long as any claim stands — it is one process, and the redirect
            # that makes it survivable belongs to it rather than to a conversation.
            named = msg.get("session_ids")
            held = server_registry.unclaim(
                [str(s) for s in named] if isinstance(named, list) and named else None)
            # Read before the send, not inside it: a getter that raises would otherwise
            # swallow the whole reply, and a client left believing the server is still
            # detached stops guarding something nothing is guarding.
            try:
                level = self.pool.worker_or_detached(
                    self._active_session_id).setting("approval_mode")
            except Exception:
                level = "manual"
            try:
                await self.ws.send(json.dumps({
                    "type": "detached", "detached": bool(held), "log": None,
                    "autonomy": level, "sessions": sorted(held), "pid": os.getpid(),
                    "setsid": False,
                }))
                # The switch itself, in the same quiet rule the attachment notice
                # uses. The button is one glyph and both of its positions look like a
                # chain: a line in the thread is what tells the user which way it just
                # went, and it says the consequence rather than only the word, since
                # "attached" on its own does not warn anybody that closing the window
                # now ends the run.
                await self._command_reply(
                    "detach",
                    ("attached — this window owns the server again, and closing it "
                     "stops the run" if not held else
                     f"attached — this window owns the server again; {len(held)} "
                     f"other conversation(s) still keep it alive"),
                    tone="quiet", transient=True)
            except Exception:
                logger.debug("_handle_detach: the client went away mid-send", exc_info=True)
                return
            logger.info("detach: taken back for %s; %d claim(s) left",
                        ", ".join(str(s) for s in named) if isinstance(named, list)
                        and named else "every conversation", len(held))
            return

        autonomy = str(msg.get("autonomy") or "manual")
        if autonomy not in ("manual", "auto", "auto_all"):
            try:
                await self.ws.send(json.dumps({
                    "type": "error",
                    "text": (f"Unknown autonomy level: {autonomy}. "
                             "Use manual, auto, or auto_all."),
                }))
            except Exception:
                logger.debug("_handle_detach: the client went away mid-send", exc_info=True)
            return

        # A claim is a *conversation's* request to be left running, so the set is the
        # conversations named — never the live agents. An agent is built on a
        # conversation's first query and released after ten minutes of idleness, and
        # the ones with no agent at that moment are exactly the ones this exists for: a
        # conversation whose work is a job MIMIR cannot see has nothing holding its
        # agent, so it is asleep when the user leaves. Built from the workers, the set
        # came out empty for it — and an empty claim is written as "not detached", so
        # the server stopped thirty seconds after the window closed, having just been
        # asked to keep going.
        #
        # The *level* still needs an agent to be applied to, and that is a separate
        # thing: one asleep gets it back from the claim when a window reattaches, and
        # from the pool-wide record when something rebuilds it meanwhile.
        session_ids = msg.get("session_ids")
        targeted: list[str] = []
        if isinstance(session_ids, list) and session_ids:
            known = {meta.id for meta in self._safe_list_sessions()}
            for sid in session_ids:
                worker = self.pool.get(sid)
                if worker is None and str(sid) not in known:
                    # Not asleep — not a conversation at all. Claiming an id nothing
                    # knows about would hold this process open for something that
                    # cannot come back to take the claim off it.
                    logger.info("detach: ignoring %s, which names no conversation", sid)
                    continue
                targeted.append(str(sid))
                if worker is None:
                    continue
                try:
                    worker.apply_setting("approval_mode", autonomy)
                except Exception:
                    logger.warning("detach: session %s refused the autonomy level %s",
                                   sid, autonomy, exc_info=True)
        else:
            self._apply_setting("approval_mode", autonomy)
            # Naming none means all of them, and "all" includes the conversation on
            # screen even before it has asked anything: a window detached the moment it
            # opened claimed nothing at all, because no agent had been built yet.
            targeted = [sid for sid, _w in self.pool.items()]
            if self._active_session_id and self._active_session_id not in targeted:
                targeted.append(self._active_session_id)

        # The agent's stdout is about to become a log file, which is not a tty but is
        # also not proof that nobody is reachable. Saying so plainly keeps the
        # interactive fallbacks — the bare ``input()`` calls behind the CLI's hooks —
        # from being attempted at all.
        #
        # Through the pool, so it is recorded as well as pushed: a detached process
        # builds agents after this point — a conversation whose own agent was released
        # gets it back when a run of its finishes — and the redirect cannot be undone,
        # so one that came up without this would reach for a terminal that is a log
        # file for the rest of the process's life.
        self.pool.set_non_interactive()

        info = detach_process(_MIMIR_DIR_WS)
        # The entry already holds the address; this only adds what has changed about
        # the process behind it, so a window that attaches later knows which
        # conversations were left running and under which level each.
        #
        # Per conversation, because that is the unit the user chose. The *process* is
        # not: one serves the whole workspace, and the redirect that makes it
        # survivable is its own — so it lives for as long as any conversation claims
        # it, and becomes mortal again when the last claim goes.
        server_registry.claim(targeted, autonomy, log=info.get("log"))

        try:
            await self.ws.send(json.dumps({
                "type": "detached",
                "detached": True,
                "log": info.get("log"),
                "autonomy": autonomy,
                "sessions": targeted,
                "pid": info.get("pid"),
                "setsid": info.get("setsid"),
            }))
            # The other half of the switch — see the take-back above. Named here as
            # what the run may do while unwatched, because that is the part of
            # detaching the user chose and the part the glyph cannot show.
            #
            # And checked against what this process would actually still have to do
            # once this window has gone, because "this run keeps working" is a claim
            # about that and nothing was testing it. A conversation whose work is a
            # job MIMIR never submitted — a raw ``sbatch`` in a shell, a submission
            # over ``ssh`` — has nothing on disk for the scan to find, so the process
            # is idle by every measure it has and stops on the idle TTL a couple of
            # hours later. Detaching into that is a night lost to a sentence that was
            # never true, and the moment to say so is the click.
            idle = self.pool.idle_report(excluding=self._sub)
            if idle["idle"]:
                # Conditional, and that is not hedging: this is the state at the click,
                # and the click can land while a turn is one tool call away from
                # submitting the very job that would keep the process alive. Said
                # flatly — "nothing is running, this will stop" — the line would be
                # false by the time it was read, which is how a true warning becomes
                # one the user learns to skip. So it says what is so now, and what
                # follows only if nothing starts. The clock itself is described by the
                # one author of that sentence, next to the TTL it is measured against.
                title = f"detached under “{autonomy}” — nothing is being watched yet"
                items = [{"label": "autonomy", "detail": autonomy}]
                note = ("No turn in flight and no run MIMIR is watching at this "
                        "moment. "
                        + idle_stop_reason(self.pool.server_idle_ttl, past=False)
                        + " Work it did not start itself is not watched: ask it to "
                        "watch the job or the process before you go.")
            else:
                title = (f"detached — this run keeps working with this window closed, "
                         f"under “{autonomy}”" if autonomy != "manual" else
                         "detached — this run keeps working with this window closed, "
                         "and parks at the first call that needs you")
                items = [{"label": "keeping it going", "detail": reason}
                         for reason in idle["reasons"]]
                note = ""
            await self._command_reply("detach", title, items=items, note=note,
                                      tone="quiet", transient=True)
        except Exception:
            logger.debug("_handle_detach: the client went away mid-send", exc_info=True)
            return
        logger.info("detach: this server is now survivable (autonomy %s over %d "
                    "conversation(s))", autonomy, len(targeted))
