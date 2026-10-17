"""What a turn leaves to review: the files it wrote (the review bar) and its checklist."""
from __future__ import annotations

import difflib
import logging
import os
from typing import Any, Callable

from ....context.todo import load_todo_items
from ..runtime import (
    todo_file_for_session,
)
from .outbox import Outbox

logger = logging.getLogger(__name__)


class Review:
    """The review bar and the checklist of one conversation.

    *session* names the conversation the worker works for — never the one on screen,
    which may be another — and *agent* the agent, None until it is built.
    """

    def __init__(self, session: Callable[[], str | None], out_q: Outbox,
                 agent: Callable[[], Any]) -> None:
        self._session = session
        self._out_q = out_q
        self._agent = agent

    def batch_status(self) -> list[dict]:
        """Per-file diffs from original to current for every snapshotted file, as the
        ``batch_status`` message carries them."""
        agent = self._agent()
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

    def push_batch_status(self) -> None:
        try:
            files = self.batch_status()
        except Exception:
            files = []
        self._out_q.put({"type": "batch_status", "files": files})

    def push_todos(self) -> None:
        try:
            items = self.todos()
            if items:
                self._out_q.put({"type": "todo", "items": items})
        except Exception:
            logger.debug("push_todos: a best-effort step failed and was skipped", exc_info=True)

    def todos(self) -> list:
        try:
            return load_todo_items(todo_file_for_session(self._session()))
        except Exception:
            logger.debug("todos: a best-effort step failed and was skipped", exc_info=True)
            return []

    def clear_todos(self) -> None:
        """Wipe this conversation's todo file."""
        try:
            todo_file = todo_file_for_session(self._session())
            if os.path.exists(todo_file):
                open(todo_file, "w").close()
        except Exception:
            logger.debug("clear_todos: a best-effort step failed and was skipped", exc_info=True)
