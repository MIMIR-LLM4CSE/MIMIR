"""Append-only JSONL log of everything a session exchanged: ``sessions/<id>/transcript.jsonl``.

Never trimmed, unlike the session JSON. Each line carries ``ts``, ``seq``, ``type`` and
the payload; token deltas are left out (their aggregates are logged).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from ...shared.state_paths import JOURNAL_FILENAME, session_file
from ..config import paths

logger = logging.getLogger(__name__)

# Streamed deltas: hundreds per turn, and each one's content is already covered by the
# aggregate event that closes the block. Logging them would make the file unreadable.
_SKIPPED_TYPES = frozenset({"token", "thinking", "context_usage", "batch_status", "todo"})


def transcript_path(session_id: str | None) -> str | None:
    """Path of a session's JSONL log, or None when there is no session to log to."""
    if not session_id:
        return None
    return session_file(JOURNAL_FILENAME, session_id, paths.STATE_DIR)


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
        """Write one event and return it with its ``seq``, the re-attach watermark (one writer per session).

        A skipped event comes back unstamped.
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
        if path is None:
            return
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
                    logger.debug("_last_seq: a best-effort step failed and was skipped", exc_info=True)
                    continue
    except Exception:
        logger.debug("_last_seq: a best-effort step failed and was skipped", exc_info=True)
        return 0
    return last


def read_since(session_id: str, after: int = 0,
               limit: int | None = None) -> tuple[list[dict], bool]:
    """Events logged after ``seq`` *after*, and whether older ones were left out.

    Streamed line by line. With *limit*, the most recent events win.
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


def nudge_counts(session_ids: list[str]) -> dict[str, int]:
    """How often each reminder category reached the model in these sessions' journals."""
    counts: dict[str, int] = {}
    for session_id in session_ids:
        for event in read_transcript(session_id):
            if event.get("type") == "nudge_injected":
                category = str(event.get("category") or "?")
                counts[category] = counts.get(category, 0) + 1
    return counts
