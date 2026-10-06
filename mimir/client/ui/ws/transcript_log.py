"""Append-only JSONL log of everything a session exchanged.

One file per session, ``<STATE_DIR>/sessions/<id>/transcript.jsonl``, next to the
``todo_list.md`` / ``todo_deps.json`` sidecars. It exists because neither of the two
lists the session JSON holds is a complete record: ``display_messages`` is what the UI
renders, and ``llm_history`` is a *window* — the front-trim in ``_handle_query`` and
``_enforce_context_budget`` mutate it in place, so the turns they drop leave no trace.
Append-only means the budget can never reach back and shorten this, which is what makes
it usable both to reconstruct the whole context and to profile a run offline.

Each line is one JSON object with ``ts`` / ``seq`` / ``type`` plus the event's own
payload. Token-level events (``token``, ``thinking`` deltas) are dropped — they would
bury the file in noise and the aggregate lands anyway at ``thinking_end`` / ``answer``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from ._ws_runtime import _MIMIR_DIR_WS

logger = logging.getLogger(__name__)

# Streamed deltas: hundreds per turn, and each one's content is already covered by the
# aggregate event that closes the block. Logging them would make the file unreadable.
_SKIPPED_TYPES = frozenset({"token", "thinking", "context_usage", "batch_status", "todo"})


def transcript_path(session_id: str | None) -> str | None:
    """Path of a session's JSONL log, or None when there is no session to log to."""
    if not session_id:
        return None
    return os.path.join(_MIMIR_DIR_WS, "sessions", session_id, "transcript.jsonl")


class TranscriptLog:
    """Append-only writer for one session's event log.

    Never raises: it is teed off the drain loop, which must keep forwarding events to
    the client even if the disk is full or read-only.
    """

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._seq = 0

    @property
    def seq(self) -> int:
        """The highest ``seq`` written so far — the watermark a client resumes from."""
        return self._seq

    def bind(self, session_id: str | None) -> None:
        """Point the log at a session, resuming its sequence numbering."""
        if session_id == self._session_id:
            return
        self._session_id = session_id
        self._seq = _last_seq(session_id)

    def append_and_stamp(self, event: dict) -> dict:
        """Write one event and return it carrying the ``seq`` it was written under.

        The stamped number is the watermark a re-attaching client resumes from, so it
        must be assigned by whoever writes the file and by nobody else — one writer per
        session. An event this log skips comes back unstamped: a stream delta is
        live-only by design, never replayed, and a ``seq`` on it would promise a
        durability it does not have.
        """
        before = self._seq
        self.append(event)
        if self._seq == before:
            return event
        event["seq"] = self._seq
        return event

    def append(self, event: dict) -> None:
        """Write one event, unless it is a streamed delta or there is no session."""
        if self._session_id is None:
            return
        etype = event.get("type")
        if etype in _SKIPPED_TYPES:
            return
        self._seq += 1
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "seq": self._seq,
            **event,
        }
        path = transcript_path(self._session_id)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            # Undo the number so the next write doesn't leave a hole in the sequence.
            self._seq -= 1
            logger.warning("Transcript log write failed for %s: %s", self._session_id, exc)


def _last_seq(session_id: str | None) -> int:
    """Highest ``seq`` already on disk, so a reconnect continues rather than restarts."""
    path = transcript_path(session_id)
    if not path or not os.path.exists(path):
        return 0
    last = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = max(last, int(json.loads(line).get("seq", 0)))
                except Exception:
                    continue
    except Exception:
        return 0
    return last


def read_since(session_id: str, after: int = 0,
               limit: int | None = None) -> tuple[list[dict], bool]:
    """Events logged after ``seq`` *after*, and whether older ones were left out.

    The read half of the re-attach watermark: a client says how far its rendered
    history goes and gets back only what it missed. Streamed line by line rather than
    through :func:`read_transcript` so a twelve-hour run is not materialised whole to
    throw most of it away.

    With *limit* set the **most recent** events win — a reconnect after a long absence
    wants the end of the run, not its beginning — and the returned flag says some were
    dropped, so the client can mark the gap rather than present a hole as the whole
    story.
    """
    path = transcript_path(session_id)
    if not path or not os.path.exists(path):
        return [], False
    events: list[dict] = []
    truncated = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    seq = int(record.get("seq", 0))
                except (TypeError, ValueError):
                    continue
                if seq <= after:
                    continue
                events.append(record)
                if limit is not None and len(events) > limit:
                    events.pop(0)
                    truncated = True
    except Exception as exc:
        logger.warning("Transcript log read failed for %s: %s", session_id, exc)
        return events, truncated
    return events, truncated


def read_transcript(session_id: str) -> list[dict]:
    """Return a session's logged events in order, skipping unparseable lines."""
    path = transcript_path(session_id)
    if not path or not os.path.exists(path):
        return []
    events: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Transcript log read failed for %s: %s", session_id, exc)
    return events
