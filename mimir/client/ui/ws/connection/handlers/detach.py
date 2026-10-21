"""Leaving the server running without this window, taking it back, stopping it.

A handler group of ``Connection``, built with its view state, its socket, the pool
and the shared ``Screen``. ``MESSAGES`` names the inbound message types it answers.
"""
from __future__ import annotations

import logging
import os

from ...process import registry as server_registry
from .....config import paths
from ...process.detach import detach as detach_process
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
        """Announce every detached run that finished while nothing watched it (``WakeRouter.sweep``).

        A task, off the handshake. A session never scanned gets a baseline instead of a backlog.
        """
        self.pool.wakes.sweep()

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
        if msg.get("enabled") is False:
            await self._take_back(msg.get("session_ids"))
            return
        autonomy = str(msg.get("autonomy") or "manual")
        if autonomy not in ("manual", "auto", "auto_all"):
            await self._send_quietly({
                "type": "error",
                "text": f"Unknown autonomy level: {autonomy}. Use manual, auto, or auto_all."})
            return

        targeted = self._claim_targets(msg.get("session_ids"), autonomy)
        # No terminal behind any agent from now on, agents built later included.
        self.pool.set_non_interactive()
        info = detach_process(paths.STATE_DIR)
        # The process lives while any claim stands.
        server_registry.claim(targeted, autonomy, log=info.get("log"))
        try:
            await self.out.send({
                "type": "detached", "detached": True, "log": info.get("log"),
                "autonomy": autonomy, "sessions": targeted, "pid": info.get("pid"),
                "setsid": info.get("setsid"),
            })
            await self._say_what_keeps_it_going(autonomy)
        except Exception:
            logger.debug("on_detach: the client went away mid-send", exc_info=True)
            return
        logger.info("detach: this server is now survivable (autonomy %s over %d "
                    "conversation(s))", autonomy, len(targeted))

    async def _send_quietly(self, payload: dict) -> None:
        try:
            await self.out.send(payload)
        except Exception:
            logger.debug("detach: the client went away mid-send", exc_info=True)

    async def _take_back(self, named: object) -> None:
        """Clear the claims of the named conversations (none named: every one).

        The redirect of the process's output stays: what keeps a server alive is that
        nobody kills it, which is a decision, not a state of the process.
        """
        names = [str(s) for s in named] if isinstance(named, list) and named else None
        held = server_registry.unclaim(names)
        # Read before the send: a getter that raises would swallow the whole reply.
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
            # The switch's glyph looks the same both ways: the line says which way it went,
            # and the consequence.
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
                    ", ".join(names) if names else "every conversation", len(held))

    def _claim_targets(self, session_ids: object, autonomy: str) -> list[str]:
        """The conversations the claim names, each live agent set to *autonomy*.

        A conversation whose agent is asleep is claimed too; an id naming no conversation
        is skipped. None named: every agent, plus the conversation on screen.
        """
        if not (isinstance(session_ids, list) and session_ids):
            self.screen.apply_setting("approval_mode", autonomy)
            targeted = [sid for sid, _w in self.pool.items()]
            if self.view.active_session_id and self.view.active_session_id not in targeted:
                targeted.append(self.view.active_session_id)
            return targeted
        known = {meta.id for meta in self._safe_list_sessions()}
        targeted = []
        for sid in session_ids:
            worker = self.pool.get(sid)
            if worker is None and str(sid) not in known:
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
        return targeted

    async def _say_what_keeps_it_going(self, autonomy: str) -> None:
        """What the process still owes without this window; work MIMIR did not start is
        invisible to it, and the server would stop on the idle TTL."""
        idle = self.pool.lifecycle.idle_report(excluding=self.view.sub)
        if idle["idle"]:
            title = f"detached under “{autonomy}” — nothing is being watched yet"
            items = [{"label": "autonomy", "detail": autonomy}]
            note = ("No turn in flight and no run MIMIR is watching at this moment. "
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
