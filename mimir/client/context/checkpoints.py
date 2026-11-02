"""Where a conversation can be rewound to: one checkpoint per message that opened a turn.

A checkpoint holds the conversation as it stood before that message (history lengths,
carried context, checklist), then the first content of every file the agent writes and
the runs it starts until the next one. Kept under the conversation's directory and
shared by every writer in this process. A file someone else changed since is never
overwritten. See docs/CLIENT_DETAILED.md, "Rewinding to a message".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import paths
from ..config.limits import CHECKPOINT_MAX_FILE_BYTES
from ...shared.atomic import write_json
from ...shared.state_paths import session_dir

logger = logging.getLogger(__name__)

CHECKPOINTS_DIRNAME = "checkpoints"
_BLOBS = "blobs"
# The digest of a file that is not there.
ABSENT = "absent"

# Read from disk at the moment of the call.
_FROM_DISK: Any = object()


def _digest(content: bytes | None) -> str:
    return ABSENT if content is None else hashlib.sha1(content).hexdigest()


def _read(path: str) -> bytes | None:
    """*path*'s bytes, None when it does not exist. Raises when it cannot be read."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def _too_big(path: str) -> bool:
    try:
        return os.path.getsize(path) > CHECKPOINT_MAX_FILE_BYTES
    except OSError:
        return False


@dataclass
class RewindPlan:
    """What rewinding to one message would do, worked out before anything is touched."""

    message_id: str
    checkpoint: dict[str, Any]
    # (path, blob digest or ABSENT): the content to put back, ABSENT meaning delete.
    restore: list[tuple[str, str]] = field(default_factory=list)
    # Changed since the agent last wrote them: left as they are.
    conflicts: list[str] = field(default_factory=list)
    # Written by the agent, but no copy of what came before was kept.
    unrestorable: list[str] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)

    def preview(self) -> dict[str, Any]:
        """The plan as the client shows it."""
        return {
            "message_id": self.message_id,
            "files": [{"path": p, "action": "delete" if d == ABSENT else "restore"}
                      for p, d in self.restore],
            "conflicts": list(self.conflicts),
            "unrestorable": list(self.unrestorable),
            "runs": [{"job_key": r.get("job_key", "?"), "kind": r.get("kind", ""),
                      "cancellable": bool((r.get("cancel_op") or {}).get("tool"))}
                     for r in self.runs],
        }


class Checkpoints:
    """The checkpoints of one conversation. One object per directory, per process."""

    _open: dict[str, "Checkpoints"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def of(cls, session_id: str | None) -> "Checkpoints | None":
        """The checkpoints of *session_id*, or None without a session."""
        if not session_id:
            return None
        directory = os.path.join(session_dir(os.path.basename(session_id), paths.STATE_DIR),
                                 CHECKPOINTS_DIRNAME)
        with cls._registry_lock:
            found = cls._open.get(directory)
            if found is None:
                found = cls._open[directory] = cls(directory)
            return found

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._lock = threading.RLock()
        self._entries: list[dict[str, Any]] | None = None

    # ── On disk ───────────────────────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        """Every checkpoint, oldest first: one small file each, so a write rewrites one."""
        if self._entries is None:
            entries = []
            try:
                names = sorted(n for n in os.listdir(self.directory) if n.endswith(".json"))
            except OSError:
                names = []
            for name in names:
                try:
                    with open(os.path.join(self.directory, name), "r", encoding="utf-8") as fh:
                        entry = json.load(fh)
                except (OSError, ValueError):
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
            self._entries = entries
        return self._entries

    def _file(self, entry: dict[str, Any]) -> str:
        return os.path.join(self.directory, f"{int(entry.get('seq') or 0):08d}.json")

    def _save(self, entry: dict[str, Any] | None = None) -> None:
        """Write *entry* (the current checkpoint by default) down."""
        entry = entry if entry is not None else self._current()
        if entry is None:
            return
        try:
            write_json(self._file(entry), entry, ensure_ascii=False)
        except Exception:
            logger.warning("checkpoints in %s would not save", self.directory, exc_info=True)

    def _remove(self, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            try:
                os.remove(self._file(entry))
            except OSError:
                pass

    def _store_blob(self, content: bytes) -> str:
        digest = _digest(content)
        path = os.path.join(self.directory, _BLOBS, digest)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(content)
            os.replace(tmp, path)
        return digest

    def _blob(self, digest: str) -> bytes:
        with open(os.path.join(self.directory, _BLOBS, digest), "rb") as fh:
            return fh.read()

    # ── Recording ─────────────────────────────────────────────────────────────

    def open(self, message_id: str, *, full_len: int, history_len: int,
             window_epoch: int, carry_context: dict[str, Any] | None = None,
             todos: list[dict[str, Any]] | None = None, todo_deps: list[list[int]] | None = None) -> None:
        """Start the checkpoint of *message_id*: the conversation as it stands before it."""
        with self._lock:
            entries = self._load()
            self._remove([e for e in entries if e.get("id") == message_id])
            entries[:] = [e for e in entries if e.get("id") != message_id]
            entries.append({
                "id": message_id,
                "seq": int(entries[-1].get("seq") or 0) + 1 if entries else 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "full_len": full_len, "history_len": history_len,
                "window_epoch": window_epoch,
                "carry_context": carry_context or {},
                "todos": list(todos or []), "todo_deps": list(todo_deps or []),
                "files": {}, "written": {}, "unknown": [], "runs": [],
            })
            self._save()

    def _current(self) -> dict[str, Any] | None:
        entries = self._load()
        return entries[-1] if entries else None

    def note_before(self, path: str, content: Any = _FROM_DISK) -> None:
        """Keep what *path* held before the agent's first write to it since the last message.

        *content* is the original when the caller observed it (a shell write, seen after
        the fact); otherwise it is read now. Never raises.
        """
        if not path:
            return
        path = os.path.abspath(path)
        try:
            with self._lock:
                current = self._current()
                if current is None or path in current["files"] or path in current["unknown"]:
                    return
                if content is _FROM_DISK:
                    if _too_big(path):
                        current["unknown"].append(path)
                        self._save()
                        return
                    content = _read(path)
                if content is not None and len(content) > CHECKPOINT_MAX_FILE_BYTES:
                    current["unknown"].append(path)
                else:
                    current["files"][path] = (ABSENT if content is None
                                              else self._store_blob(content))
                self._save()
        except Exception:
            logger.debug("note_before: %s could not be kept", path, exc_info=True)
            self.note_unknown(path)

    def note_unknown(self, path: str) -> None:
        """*path* was written by the agent and what it held before is not known."""
        path = os.path.abspath(path)
        with self._lock:
            current = self._current()
            if current is None or path in current["files"] or path in current["unknown"]:
                return
            current["unknown"].append(path)
            self._save()

    def note_written(self, path: str) -> None:
        """Record what the agent left in *path*, to tell a later change by someone else. Never raises."""
        if not path:
            return
        path = os.path.abspath(path)
        try:
            with self._lock:
                current = self._current()
                if current is None:
                    return
                current["written"][path] = _digest(_read(path))
                self._save()
        except Exception:
            logger.debug("note_written: %s could not be read", path, exc_info=True)

    def note_run(self, descriptor: dict[str, Any]) -> None:
        """A run started during the current checkpoint's turns."""
        if not isinstance(descriptor, dict):
            return
        with self._lock:
            current = self._current()
            if current is None:
                return
            key = descriptor.get("job_key")
            if any(r.get("job_key") == key for r in current["runs"]):
                return
            current["runs"].append(dict(descriptor))
            self._save()

    # ── Rewinding ─────────────────────────────────────────────────────────────

    def has(self, message_id: str) -> bool:
        with self._lock:
            return any(e.get("id") == message_id for e in self._load())

    def ids(self) -> list[str]:
        with self._lock:
            return [str(e.get("id")) for e in self._load()]

    def plan(self, message_id: str) -> RewindPlan | None:
        """What rewinding to *message_id* would put back and stop, or None when it has no checkpoint."""
        with self._lock:
            entries = self._load()
            index = next((i for i, e in enumerate(entries) if e.get("id") == message_id), None)
            if index is None:
                return None
            since = entries[index:]
            plan = RewindPlan(message_id=message_id, checkpoint=dict(entries[index]))
            # The earliest checkpoint that saw a path holds what it was before the message.
            first: dict[str, str | None] = {}
            for entry in since:
                for path, digest in entry.get("files", {}).items():
                    first.setdefault(path, digest)
                for path in entry.get("unknown", []):
                    first.setdefault(path, None)
            for path in sorted(first):
                original = first[path]
                if original is None:
                    plan.unrestorable.append(path)
                    continue
                try:
                    now = _digest(_read(path))
                except OSError:
                    plan.unrestorable.append(path)
                    continue
                if now == original:
                    continue
                left = next((e["written"][path] for e in reversed(since)
                             if path in e.get("written", {})), None)
                if left is not None and left != now:
                    plan.conflicts.append(path)
                    continue
                plan.restore.append((path, original))
            for entry in since:
                plan.runs.extend(entry.get("runs", []))
            return plan

    def restore(self, path: str, digest: str) -> bool:
        """Put *path* back to blob *digest* (ABSENT: remove it). False when it could not be done."""
        try:
            if digest == ABSENT:
                if os.path.lexists(path):
                    os.remove(path)
                return True
            content = self._blob(digest)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.mimir-rewind.tmp"
            with open(tmp, "wb") as fh:
                fh.write(content)
            try:
                os.chmod(tmp, os.stat(path).st_mode)
            except OSError:
                pass
            os.replace(tmp, path)
            return True
        except OSError:
            logger.warning("rewind: %s could not be put back", path, exc_info=True)
            return False

    def drop_from(self, message_id: str) -> None:
        """Forget the checkpoint of *message_id* and every later one, and the blobs only they held."""
        with self._lock:
            entries = self._load()
            index = next((i for i, e in enumerate(entries) if e.get("id") == message_id), None)
            if index is None:
                return
            self._remove(entries[index:])
            del entries[index:]
            kept = {d for e in entries for d in e.get("files", {}).values()}
            blobs = os.path.join(self.directory, _BLOBS)
            try:
                names = os.listdir(blobs)
            except OSError:
                return
            for name in names:
                if name not in kept:
                    try:
                        os.remove(os.path.join(blobs, name))
                    except OSError:
                        pass
