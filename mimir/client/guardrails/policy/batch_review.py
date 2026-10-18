"""Writes held for one review at the end of the turn, and the means to undo them.

In batch mode a write runs at once but is queued here; the first content seen of each
file is snapshotted, so the turn's whole change can be shown as one diff and taken
back as a whole. A revert only ever undoes what the user was shown: what a file gained
since comes from somewhere this agent cannot see — under concurrent sessions, usually
another conversation.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from typing import Callable


class BatchReview:
    def __init__(self) -> None:
        self.pending: list[dict] = []
        # path -> original content (None: the file did not exist before the batch).
        self.snapshots: dict[str, str | None] = {}
        # Digest of each reviewed file as it stood when its diff was last shown.
        self._reviewed: dict[str, str] = {}

    @property
    def has_pending(self) -> bool:
        return bool(self.pending)

    def queue(self, tool_name: str, server_name: str, arguments: dict) -> None:
        """Hold a write for the end-of-turn review."""
        self.pending.append(
            {"tool_name": tool_name, "server_name": server_name, "arguments": arguments})

    def record_snapshot(self, path: str) -> None:
        """Capture the current content of *path* before a write modifies it.

        Only the first snapshot per path is kept so subsequent edits within
        the same batch don't overwrite the original baseline.
        """
        if path in self.snapshots:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                self.snapshots[path] = fh.read()
        except FileNotFoundError:
            self.snapshots[path] = None  # file will be created by this batch
        except OSError:
            pass  # can't snapshot; skip gracefully

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()

    def note_reviewed(self, path: str, content: str) -> None:
        """Record *content* as the state of *path* the user was just shown.

        Called while the review diff is built, the one moment at which what is on disk and
        what the user is looking at are known to be the same thing.
        """
        self._reviewed[path] = self._digest(content)

    def reviewed_matches(self, path: str) -> bool | None:
        """Whether *path* on disk is still what was reviewed. None = cannot tell.

        None when nothing was recorded, or the file cannot be read: the caller then
        proceeds, like every other best-effort check in this module — a guard that cannot
        read must not block the user's own undo.
        """
        expected = self._reviewed.get(path)
        if expected is None:
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return self._digest(fh.read()) == expected
        except FileNotFoundError:
            # Gone since the review: restoring recreates it from a baseline nobody asked
            # for, so this counts as moved.
            return False
        except OSError:
            return None

    def forget_reviewed(self, path: str | None = None) -> None:
        """Drop the reviewed digest for *path*, or all of them."""
        if path is None:
            self._reviewed.clear()
        else:
            self._reviewed.pop(path, None)

    def flush(
        self,
        output_fn: Callable[[str], None] | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> bool:
        """Show a combined diff of all batch edits and ask for a single approval.

        Returns True if the user accepted, False if they rejected (in which case
        all snapshotted files are restored to their pre-batch state).
        """
        output_fn = output_fn or print
        input_fn = input_fn or input

        if not self.pending:
            return True

        output_fn(
            f"\n📝 Batch review — {len(self.pending)} write operation(s) performed:"
        )

        # Per-operation summary
        for i, edit in enumerate(self.pending, 1):
            args_display = {
                k: v
                for k, v in edit["arguments"].items()
                if k not in ("new_text", "content")
            }
            new_val = edit["arguments"].get("new_text") or edit["arguments"].get("content", "")
            if new_val:
                preview = new_val[:120].replace("\n", "↵")
                args_display["new_text"] = preview + ("…" if len(new_val) > 120 else "")
            output_fn(
                f"\n  [{i}] {edit['tool_name']}  (server: {edit['server_name']})\n"
                + json.dumps(args_display, indent=4, ensure_ascii=False, default=str)
            )

        # Unified diffs for every snapshotted/modified file
        for path, original in self.snapshots.items():
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    current = fh.read()
            except OSError:
                current = ""

            before_lines = original.splitlines(keepends=True) if original is not None else []
            after_lines = current.splitlines(keepends=True)

            if before_lines == after_lines:
                continue

            rel = os.path.relpath(path)
            diff = "".join(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                    n=3,
                )
            )
            output_fn(f"\n{diff}" if diff else f"\n(binary change: {rel})")

        # Single approval prompt
        choice = input_fn(
            "\nApply all changes? [y]es / [n]o (undo all): "
        ).strip().lower()
        approved = choice in {"y", "yes"}

        if not approved:
            for path, original in self.snapshots.items():
                try:
                    if original is None:
                        os.remove(path)
                    else:
                        with open(path, "w", encoding="utf-8") as fh:
                            fh.write(original)
                except OSError as exc:
                    output_fn(f"  ⚠️  Could not restore {path}: {exc}")
            output_fn("  ↩  Changes reverted.")
        else:
            output_fn("  ✔  Changes applied.")

        self.pending.clear()
        self.snapshots.clear()
        return approved

    def revert(self, output_fn: Callable[[str], None] | None = None) -> list[str]:
        """Restore every snapshotted file (deleting the ones the agent created); return their paths."""
        output_fn = output_fn or print

        if not self.snapshots:
            output_fn("Nothing to undo — no file snapshots are available for this session.")
            return []

        reverted: list[str] = []
        for path, original in self.snapshots.items():
            try:
                if original is None:
                    if os.path.exists(path):
                        os.remove(path)
                        output_fn(f"  ↩  Deleted (newly created): {path}")
                    reverted.append(path)
                else:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(original)
                    output_fn(f"  ↩  Restored: {path}")
                    reverted.append(path)
            except OSError as exc:
                output_fn(f"  ⚠️  Could not restore {path}: {exc}")

        self.pending.clear()
        self.snapshots.clear()
        return reverted
