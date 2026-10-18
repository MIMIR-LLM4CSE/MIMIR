"""Leaving the server running without this window, taking it back, stopping it.

A handler group of ``Connection``, built with its view state, its socket, the pool
and the shared ``Screen``. ``MESSAGES`` names the inbound message types it answers.
"""
from __future__ import annotations

import logging
import os

from ...server import registry as server_registry
from .....config import paths
from ...server.detach import detach as detach_process
from .....session.jobs_on_disk import establish_baseline, has_baseline, scan_all_sessions
from ...pool.lifecycle import idle_stop_reason
from ...pool.pool import AgentPool
from ..screen import Screen
from ..view import Sender, ViewState
from .base import Handlers

logger = logging.getLogger(__name__)


class DetachHandlers(Handlers):
    MESSAGES: dict[str, str] = {
        "detach": "on_detach",
        "shutdown": "on_shutdown",
    }

    def __init__(self, view: ViewState, out: Sender, pool: AgentPool, screen: Screen) -> None:
        super().__init__(view, out, pool, screen)
        self.store = pool.conversations.store

    async def report_ended_jobs(self) -> None:
        """Announce every detached run that finished while nothing watched it, through the wake router.

        A task, off the handshake. Marked on disk as announced; a session never scanned gets a
        baseline instead of a backlog.
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

    async def restore_detached_autonomy(self) -> None:
        """Take back a run left going: drop the claims, restore each conversation's level.

        The level is read from the claim before it goes; it lives only as long as the run.
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
        level = held.get(self.view.active_session_id or "") or sorted(held.values())[-1]
        self.screen.apply_setting("approval_mode", level)
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
            await self.out.send({
                "type": "detached", "detached": False, "log": None,
                "autonomy": level, "sessions": [], "pid": os.getpid(),
                "setsid": False,
            })
            await self.out.send({"type": "approval_mode", "mode": level})
            # Rendered as a quiet, transient rule: where the user rejoined and under what. It opens
            # with "attached" (the toggle's word), then the consequence: closing this window now ends
            # the run. The level is named only if not manual, the count only if several.
            under = f", under “{level}”" if level != "manual" else ""
            title = (
                f"attached — this conversation kept working with no window open"
                f"{under}; closing this window now stops it"
                if len(held) == 1 else
                f"attached — {len(held)} conversations kept working with no window "
                f"open, this one{under or ' at manual'}; closing this window now "
                f"stops them")
            await self.screen.command_reply("detach", title, tone="quiet", transient=True)
        except Exception:
            logger.debug("_restore_detached_autonomy: the client went away mid-send", exc_info=True)
            return
        logger.info("reattach: %d claim(s) taken back; autonomy restored to %s from "
                    "the detached run", len(held), level)

    async def on_shutdown(self, msg: dict) -> None:
        """Stop the server; refused with the reasons while work runs, unless ``force``."""
        report = self.pool.lifecycle.idle_report()
        force = bool(msg.get("force"))
        if not report["idle"] and not force:
            try:
                await self.out.send({
                    "type": "shutdown_refused",
                    "reasons": report["reasons"],
                })
            except Exception:
                logger.debug("on_shutdown: the client went away mid-send", exc_info=True)
            return
        try:
            await self.out.send({
                "type": "shutting_down",
                "forced": force,
                "reasons": report["reasons"],
            })
        except Exception:
            logger.debug("on_shutdown: the client went away mid-send", exc_info=True)
        logger.info("shutdown: requested by the client (forced=%s; %s)", force,
                    "; ".join(report["reasons"]) or "nothing was running")
        self.pool.lifecycle.request_stop()

    def _safe_list_sessions(self) -> list:
        """The stored conversations, or [] when the store cannot be read."""
        try:
            return self.store.list_sessions()
        except Exception:
            logger.warning("detach: the stored conversations could not be listed",
                           exc_info=True)
            return []

    async def on_detach(self, msg: dict) -> None:
        """"Continue without me": make this process survivable and set its autonomy.

        The level is required and applied to the named conversations (none: all of them, and
        recorded pool-wide). ``enabled: false`` takes the claim back.
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
                    self.view.active_session_id).setting("approval_mode")
            except Exception:
                level = "manual"
            try:
                await self.out.send({
                    "type": "detached", "detached": bool(held), "log": None,
                    "autonomy": level, "sessions": sorted(held), "pid": os.getpid(),
                    "setsid": False,
                })
                # The switch itself, in the same quiet rule the attachment notice
                # uses. The button is one glyph and both of its positions look like a
                # chain: a line in the thread is what tells the user which way it just
                # went, and it says the consequence rather than only the word, since
                # "attached" on its own does not warn anybody that closing the window
                # now ends the run.
                await self.screen.command_reply(
                    "detach",
                    ("attached — this window owns the server again, and closing it "
                     "stops the run" if not held else
                     f"attached — this window owns the server again; {len(held)} "
                     f"other conversation(s) still keep it alive"),
                    tone="quiet", transient=True)
            except Exception:
                logger.debug("on_detach: the client went away mid-send", exc_info=True)
                return
            logger.info("detach: taken back for %s; %d claim(s) left",
                        ", ".join(str(s) for s in named) if isinstance(named, list)
                        and named else "every conversation", len(held))
            return

        autonomy = str(msg.get("autonomy") or "manual")
        if autonomy not in ("manual", "auto", "auto_all"):
            try:
                await self.out.send({
                    "type": "error",
                    "text": (f"Unknown autonomy level: {autonomy}. "
                             "Use manual, auto, or auto_all."),
                })
            except Exception:
                logger.debug("on_detach: the client went away mid-send", exc_info=True)
            return

        # A claim belongs to the conversations named, never to the live agents: one whose agent
        # is asleep is exactly the case detaching exists for. Its level comes back from the claim
        # on reattach, or from the pool-wide record on a rebuild.
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
            self.screen.apply_setting("approval_mode", autonomy)
            # Naming none means all of them, and "all" includes the conversation on
            # screen even before it has asked anything: a window detached the moment it
            # opened claimed nothing at all, because no agent had been built yet.
            targeted = [sid for sid, _w in self.pool.items()]
            if self.view.active_session_id and self.view.active_session_id not in targeted:
                targeted.append(self.view.active_session_id)

        # No terminal behind any agent from now on, recorded in the pool so agents built later
        # know it too.
        self.pool.set_non_interactive()

        info = detach_process(paths.STATE_DIR)
        # Record which conversations were left running and under which level; the process lives
        # while any claim stands.
        server_registry.claim(targeted, autonomy, log=info.get("log"))

        try:
            await self.out.send({
                "type": "detached",
                "detached": True,
                "log": info.get("log"),
                "autonomy": autonomy,
                "sessions": targeted,
                "pid": info.get("pid"),
                "setsid": info.get("setsid"),
            })
            # Checked against what the process would still have to do without this window: work
            # MIMIR never started is invisible, and the server would stop on the idle TTL.
            idle = self.pool.lifecycle.idle_report(excluding=self.view.sub)
            if idle["idle"]:
                # Conditional on purpose: the state at the click, and what follows only if nothing starts.
                title = f"detached under “{autonomy}” — nothing is being watched yet"
                items = [{"label": "autonomy", "detail": autonomy}]
                note = ("No turn in flight and no run MIMIR is watching at this "
                        "moment. "
                        + idle_stop_reason(self.pool.lifecycle.server_idle_ttl, past=False)
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
            await self.screen.command_reply("detach", title, items=items, note=note,
                                      tone="quiet", transient=True)
        except Exception:
            logger.debug("on_detach: the client went away mid-send", exc_info=True)
            return
        logger.info("detach: this server is now survivable (autonomy %s over %d "
                    "conversation(s))", autonomy, len(targeted))
