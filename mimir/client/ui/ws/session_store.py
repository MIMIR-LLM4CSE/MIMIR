"""Persistent session store for MimirAgent.

Each session is saved as a JSON file under .mimir/sessions/<id>.json.

Session JSON schema:
{
  "id": "uuid4",
  "title": "auto-generated or user-set",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "preview": "first 80 chars of first user message",
  "summary": "one-sentence description of what was done in the session",
  "summary_msgs": 0,          # display-message count the summary was generated from
  "summary_version": 0,       # session_summary.SUMMARY_VERSION that produced it
  "title_custom": false,      # true once the user renamed the session by hand
  "llm_history": [{"role": "...", "content": "..."}],      # working window sent to the LLM
  "llm_history_full": [...],  # same, never trimmed — the archive a resume falls back to
                              #   (a window that carries a compaction summary is what a
                              #   resume reloads instead; see _Session._load_session)
  "display_messages": [...],  # serialised UI ChatMessage objects
  "carry_context": {...},     # MimirAgent._carry_context
  "todos": [{"text": "...", "done": false}],
  "pending_interaction": {...} | null  # the card a deferred turn waits on, and what
                                       #   its answer resumes (query_engine.deferral)
}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from ...config.constants import STATE_DIR

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_dir() -> str:
    """Return (and create if needed) the central <STATE_DIR>/sessions/ directory."""
    base = os.path.join(STATE_DIR, "sessions")
    os.makedirs(base, exist_ok=True)
    return base


# ── Sub-agents ────────────────────────────────────────────────────────────────
# A sub-agent writes under a session of its own, stored *inside* its parent's sidecar
# directory: sessions/<parent>/subagents/<child>/. Two things follow for free — the
# front-end session list never shows one (it enumerates the <id>.json files at the root
# of sessions/, and a sub-session writes only directories), and deleting a session takes
# its children with it (delete_session drops the whole sidecar).

_SUBAGENTS_DIRNAME = "subagents"

# What a card carries of the child's answer. The whole answer is on disk and the detail
# view serves it whole; the list goes out as one message holding every card of the
# session, and a handful of multi-kilobyte answers in it is a payload nobody reads.
_CARD_ANSWER_CHARS = 400


def _subagents_dir(session_id: str) -> str:
    return os.path.join(_sessions_dir(), os.path.basename(session_id), _SUBAGENTS_DIRNAME)


def list_subagents(session_id: str) -> list[dict]:
    """The sub-agents this session spawned, newest first.

    Each is the card the spawn server left (``subagent.json``): what it was asked, what
    it was given, how it ended. A directory without one is a child that died before
    writing anything, reported as such rather than hidden — an axis that vanished is
    precisely what the user needs to see.

    The answer is clipped here and only here: it is the one field that can run to
    kilobytes, this list carries every card of the session at once, and the panel shows
    an answer only in the detail view, which reads it through :func:`read_subagent`.
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
        with open(os.path.join(directory, "activity.jsonl"), "r", encoding="utf-8") as f:
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
    card = {"id": os.path.basename(sub_id), "state": "unknown"}
    try:
        with open(os.path.join(directory, "subagent.json"), "r", encoding="utf-8") as f:
            card.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    card["id"] = os.path.basename(sub_id)
    try:
        with open(os.path.join(directory, "todo_list.md"), "r", encoding="utf-8") as f:
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
    summary_version: int = 0   # session_summary.SUMMARY_VERSION that produced it
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
        )


# ── Store ──────────────────────────────────────────────────────────────────────

class SessionStore:
    """Reads and writes sessions to .mimir/sessions/<id>.json."""

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
        sdir = _sessions_dir()
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
        """Atomically write session to disk (write tmp then rename)."""
        sdir = _sessions_dir()
        target = os.path.join(sdir, f"{session.id}.json")
        tmp = target + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, target)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    def delete_session(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)
        # Drop the sidecar directory too (todo_list.md, todo_deps.json,
        # transcript.jsonl) — otherwise a deleted session leaves its log behind and a
        # recycled id would append to a stranger's transcript.
        sidecar = os.path.join(_sessions_dir(), os.path.basename(session_id))
        if os.path.isdir(sidecar):
            shutil.rmtree(sidecar, ignore_errors=True)

    def session_exists(self, session_id: str) -> bool:
        return os.path.exists(self._path(session_id))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> str:
        # Guard against path traversal.
        safe_id = os.path.basename(session_id)
        return os.path.join(_sessions_dir(), f"{safe_id}.json")
