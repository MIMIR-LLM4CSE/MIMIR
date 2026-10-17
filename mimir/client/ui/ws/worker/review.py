"""What a turn leaves to review: the files it wrote (the review bar) and its checklist."""
from __future__ import annotations

import difflib
import logging
import os
from ....context.todo import load_todo_items
from ..runtime import (
    todo_file_for_session,
)

logger = logging.getLogger(__name__)


class WorkerReview:
        # batch_status is sent from the drain loop's answer handler instead, so it
        # reflects the current snapshot — a batch_status queued here would lose the
        # race with an in-flight batch_review_accept.

    def _build_batch_status(self) -> list[dict]:
        """Compute per-file accumulated diffs from original → current for every
        snapshotted file.  Returns a list of diff entries suitable for the
        ``batch_status`` WebSocket message.
        """

        agent = self._agent
        if agent is None:
            return []
        snapshots = agent.approvals.batch.snapshots
        files: list[dict] = []
        for path, original in list(snapshots.items()):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    current = fh.read()
            except OSError:
                current = ""
            # What the user is about to be shown, recorded so a later revert can tell
            # whether the file still holds it. Anything else on disk by then was written
            # by something outside this review — another session, or the user's editor.
            agent.approvals.batch.note_reviewed(path, current)
            before_lines = original.splitlines(keepends=True) if original is not None else []
            after_lines  = current.splitlines(keepends=True)
            if before_lines == after_lines:
                continue
            rel = os.path.relpath(path)
            diff_lines = list(difflib.unified_diff(
                before_lines, after_lines,
                fromfile="a/" + rel, tofile="b/" + rel, n=3,
            ))
            if diff_lines:
                entry: dict = {"file": rel, "patch": "".join(diff_lines)}
                if original is None:
                    entry["is_new"] = True
                files.append(entry)
        return files

    def _push_batch_status(self) -> None:
        try:
            files = self._build_batch_status()
        except Exception:
            files = []
        self.out_q.put({"type": "batch_status", "files": files})

    def _push_todos(self) -> None:
        try:
            items = self._load_todos()
            if items:
                self.out_q.put({"type": "todo", "items": items})
        except Exception:
            logger.debug("_push_todos: a best-effort step failed and was skipped", exc_info=True)

    def _own_session(self) -> str | None:
        """The conversation this worker works for — never the one on screen.

        It fell back to the session on screen once, for a worker built before any
        session was known; one with a session of its own reading the screen instead
        wrote another conversation's checklist the moment the user looked elsewhere.
        A worker with no session (the pre-agent stand-in) has none.
        """
        return self.session_id

    def _load_todos(self) -> list:
        try:
            return load_todo_items(todo_file_for_session(self._own_session()))
        except Exception:
            logger.debug("_load_todos: a best-effort step failed and was skipped", exc_info=True)
            return []

    def _clear_todos(self) -> None:
        """Wipe this worker's own session's todo file."""
        try:
            todo_file = todo_file_for_session(self._own_session())
            if os.path.exists(todo_file):
                open(todo_file, "w").close()
        except Exception:
            logger.debug("_clear_todos: a best-effort step failed and was skipped", exc_info=True)
