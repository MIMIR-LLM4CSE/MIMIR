"""The persistent session store: one JSON file per session under ``<STATE_DIR>/sessions/``.

The schema is in CLIENT_DETAILED.md, "The session file".
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from ..config import paths
from ...shared.atomic import write_json
from ...shared.state_paths import (
    ACTIVITY_FILENAME,
    SUBAGENTS_DIRNAME,
    TODO_FILENAME,
    sessions_root,
)

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_dir() -> str:
    """Return (and create if needed) the central <STATE_DIR>/sessions/ directory."""
    base = sessions_root(paths.STATE_DIR)
    os.makedirs(base, exist_ok=True)
    return base


# ── Sub-agents ────────────────────────────────────────────────────────────────
# A sub-agent writes under a session of its own, stored *inside* its parent's sidecar
# directory: sessions/<parent>/subagents/<child>/. Two things follow for free — the
# front-end session list never shows one (it enumerates the <id>.json files at the root
# of sessions/, and a sub-session writes only directories), and deleting a session takes
# its children with it (delete_session drops the whole sidecar).

# What a card carries of the child's answer. The whole answer is on disk and the detail
# view serves it whole; the list goes out as one message holding every card of the
# session, and a handful of multi-kilobyte answers in it is a payload nobody reads.
_CARD_ANSWER_CHARS = 400


def _subagents_dir(session_id: str) -> str:
    return os.path.join(_sessions_dir(), os.path.basename(session_id), SUBAGENTS_DIRNAME)


def _still_running(card: dict) -> bool:
    """Whether a card saying "running" belongs to a process that still exists.

    Erring towards alive: no pid, or another user's process, counts as running.
    """
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


def list_subagents(session_id: str) -> list[dict]:
    """The sub-agents this session spawned, newest first, from their ``subagent.json`` cards.

    A directory without a card is reported as dead; a "running" card whose process is gone
    as abandoned. Answers are clipped here (the detail view reads them whole).
    """
    base = _subagents_dir(session_id) if session_id else ""
    if not base or not os.path.isdir(base):
        return []
    cards: list[dict] = []
    for name in os.listdir(base):
        directory = os.path.join(base, name)
        if not os.path.isdir(directory):
            continue
        card = {"id": name, "state": "unknown"}
        try:
            with open(os.path.join(directory, "subagent.json"), "r", encoding="utf-8") as f:
                card.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
        card["id"] = name           # the directory is the identity, not the file
        # Said to be working, by a process that is gone: it was abandoned mid-run, and
        # nothing will ever write its ending. Reported as such rather than left running
        # for the life of the session.
        if card.get("state") == "running" and not _still_running(card):
            card["state"] = "abandoned"
        answer = card.get("answer") or ""
        if len(answer) > _CARD_ANSWER_CHARS:
            card["answer"] = answer[:_CARD_ANSWER_CHARS] + "…"
        cards.append(card)
    cards.sort(key=lambda c: c.get("started_at", ""), reverse=True)
    return cards


def _activity_tail(directory: str, limit: int = 120) -> list[dict]:
    """The last rows of a sub-agent's activity log, oldest first.

    The log is written line by line by another process, possibly mid-write: a line
    that does not parse is dropped rather than failing the read, because a half-written
    last row is the normal state of a child that is still working.
    """
    rows: list[dict] = []
    try:
        with open(os.path.join(directory, ACTIVITY_FILENAME), "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_subagent(session_id: str, sub_id: str) -> dict:
    """One sub-agent's card, the todo list it kept, and what it has been doing, or ``{}``.

    ``sub_id`` is confined with ``basename`` the way session ids are: it arrives from
    the front end, and nothing read here is meant to sit outside this session's own
    sub-agent directory.
    """
    if not session_id or not sub_id:
        return {}
    directory = os.path.join(_subagents_dir(session_id), os.path.basename(sub_id))
    if not os.path.isdir(directory):
        return {}
    card: dict[str, Any] = {"id": os.path.basename(sub_id), "state": "unknown"}
    try:
        with open(os.path.join(directory, "subagent.json"), "r", encoding="utf-8") as f:
            card.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    card["id"] = os.path.basename(sub_id)
    try:
        with open(os.path.join(directory, TODO_FILENAME), "r", encoding="utf-8") as f:
            card["todo"] = f.read()
    except OSError:
        card["todo"] = ""
    card["activity"] = _activity_tail(directory)
    return card


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str = ""
    summary: str = ""
    summary_version: int = 0
    title_custom: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FullSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str = ""
    summary: str = ""          # one-sentence description of what the session did
    summary_msgs: int = 0      # display-message count the summary was generated from
    summary_version: int = 0   # session.summary.SUMMARY_VERSION that produced it
    title_custom: bool = False # True once the user renamed the session by hand
    llm_history: list[dict] = field(default_factory=list)
    # The working window, as trimmed by the context budget, and the untrimmed record it
    # was cut from. Only the second survives a long session, so a resume starts there.
    llm_history_full: list[dict] = field(default_factory=list)
    display_messages: list[dict] = field(default_factory=list)
    carry_context: dict = field(default_factory=dict)
    todos: list[dict] = field(default_factory=list)
    todo_deps: list[list[int]] = field(default_factory=list)
    # What a turn of this conversation was waiting on when the user left it: the
    # card to put back and the calls its answer resumes (query_engine.deferral).
    pending_interaction: dict | None = None
    # The context mode the session was running in. Read on resume to size the window
    # before this conversation has an agent to ask.
    context_mode: str = "full"
    # Bumped on every save. Two things write a session file now — a connected
    # ``Connection`` and, when none is connected, the pump's turn committer — and the
    # revision is what lets a save notice it is built on a copy someone else has
    # already superseded, instead of overwriting it with older history.
    rev: int = 0
    # How far into this session's ``transcript.jsonl`` the stored ``display_messages``
    # already go. The client assembles the rich transcript — tool rows, reasoning
    # panels, diff cards — so this is its own statement of what it has rendered, and it
    # is what a re-attach replays *from*. Stored rather than kept per connection: a
    # fresh window, or another machine, has to be able to resume too.
    rendered_seq: int = 0

    def meta(self) -> SessionMeta:
        return SessionMeta(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            preview=self.preview,
            summary=self.summary,
            summary_version=self.summary_version,
            title_custom=self.title_custom,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FullSession":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            preview=data.get("preview", ""),
            summary=data.get("summary", ""),
            summary_msgs=data.get("summary_msgs", 0),
            summary_version=data.get("summary_version", 0),
            title_custom=bool(data.get("title_custom", False)),
            llm_history=data.get("llm_history", []),
            llm_history_full=data.get("llm_history_full", []),
            display_messages=data.get("display_messages", []),
            carry_context=data.get("carry_context", {}),
            todos=data.get("todos", []),
            todo_deps=data.get("todo_deps", []),
            pending_interaction=data.get("pending_interaction") or None,
            # A session file without the field is full-mode, the mode an agent starts in.
            context_mode=data.get("context_mode") or "full",
            rev=int(data.get("rev") or 0),
            rendered_seq=int(data.get("rendered_seq") or 0),
        )


class StaleSessionWrite(RuntimeError):
    """A save whose base revision is behind what is already on disk."""


def _stored_rev(path: str) -> int:
    """The revision in the file at *path*, or 0 when there is no readable one."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("rev") or 0)
    except Exception:
        logger.debug("_stored_rev: a best-effort step failed and was skipped", exc_info=True)
        return 0


# ── Store ──────────────────────────────────────────────────────────────────────

class SessionStore:
    """Reads and writes sessions to <STATE_DIR>/sessions/<id>.json.

    *root* puts the ``sessions`` directory elsewhere — a test's own directory. Left
    out, ``STATE_DIR`` is read at each call, so patching it redirects every store.
    """

    def __init__(self, root: str | None = None) -> None:
        self._root = root

    def _dir(self) -> str:
        if self._root is None:
            return _sessions_dir()
        base = sessions_root(self._root)
        os.makedirs(base, exist_ok=True)
        return base

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def new_session(self) -> FullSession:
        now = _now_iso()
        return FullSession(
            id=str(uuid.uuid4()),
            title="",
            created_at=now,
            updated_at=now,
        )

    def list_sessions(self) -> list[SessionMeta]:
        """Return sessions sorted newest-first (by updated_at)."""
        metas: list[SessionMeta] = []
        sdir = self._dir()
        for fname in os.listdir(sdir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(sdir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metas.append(SessionMeta(
                    id=data.get("id", fname[:-5]),
                    title=data.get("title", ""),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    preview=data.get("preview", ""),
                    summary=data.get("summary", ""),
                    summary_version=data.get("summary_version", 0),
                    title_custom=bool(data.get("title_custom", False)),
                ))
            except Exception as exc:
                logger.warning("Failed to read session %s: %s", fname, exc)
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas

    def load_session(self, session_id: str) -> FullSession:
        path = self._path(session_id)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FullSession.from_dict(data)

    def save_session(self, session: FullSession) -> None:
        """Write *session* atomically; raise :class:`StaleSessionWrite` if the file moved past its revision."""
        sdir = self._dir()
        target = os.path.join(sdir, f"{session.id}.json")
        on_disk = _stored_rev(target)
        if on_disk > session.rev:
            raise StaleSessionWrite(
                f"session {session.id} moved on (disk rev {on_disk} > {session.rev})")
        session.rev = on_disk + 1
        try:
            write_json(target, session.to_dict(), ensure_ascii=False, indent=2,
                              default=str)
        except Exception:
            session.rev = on_disk
            raise

    def delete_session(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)
        # Drop the sidecar directory too (todo_list.md, todo_deps.json,
        # transcript.jsonl) — otherwise a deleted session leaves its log behind and a
        # recycled id would append to a stranger's transcript.
        sidecar = os.path.join(self._dir(), os.path.basename(session_id))
        if os.path.isdir(sidecar):
            shutil.rmtree(sidecar, ignore_errors=True)

    def session_exists(self, session_id: str) -> bool:
        return os.path.exists(self._path(session_id))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> str:
        # Guard against path traversal.
        safe_id = os.path.basename(session_id)
        return os.path.join(self._dir(), f"{safe_id}.json")
