"""What one query leaves for the next in the same session: the carry context.

Only accumulative fields are carried (what was read, searched, found to exist, the
skills already loaded); per-query state (workflow phase, nudge counts, denials) starts
clean every query. A read is carried with the file's mtime at the end of the query, so
a file changed on disk since is read again rather than trusted.
"""
from __future__ import annotations

import os
from typing import Any

from .execution_context import (
    CARRY,
    ExecutionContext,
    carry_context_from_json,
    carry_context_to_json,
    carry_path_fields,
    fields_with,
)
from .paths import absolute_workspace_path


def _mirrored_fields() -> tuple[str, ...]:
    """The execution-context fields mirrored into the carry and back (the CARRY trait):
    sets are unioned, dicts merged, flags kept once raised."""
    return fields_with(CARRY)


class CarryContext:
    def __init__(self, fields: dict[str, Any] | None = None) -> None:
        self.fields: dict[str, Any] = fields if fields is not None else {}

    # ── Session persistence ────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        return carry_context_to_json(self.fields)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CarryContext:
        return cls(carry_context_from_json(data or {}))

    # ── What a sub-agent hands back ────────────────────────────────────────────

    def files_read(self) -> list[str]:
        return sorted(self.fields.get("read_files") or [])

    def files_written(self) -> list[str]:
        """The files the last query wrote."""
        return sorted(self.fields.get("last_query_written_files") or [])

    # ── Into a query and out of it ─────────────────────────────────────────────

    def apply(self, execution_context: ExecutionContext) -> None:
        """Seed a fresh query's execution context with what the session knows.

        A carried read whose file changed on disk since is dropped first, so the
        policy does not count it as a read of the current content.
        """
        carry = self.fields
        prior_reads: set[str] = carry.get("read_files", set())
        if prior_reads:
            read_mtimes: dict[str, float] = carry.get("_read_mtimes", {})
            fresh_reads: set[str] = set()
            for p in prior_reads:
                recorded_mtime = read_mtimes.get(p)
                if recorded_mtime is None:
                    fresh_reads.add(p)
                    continue
                try:
                    if os.path.getmtime(absolute_workspace_path(p)) <= recorded_mtime:
                        fresh_reads.add(p)
                except OSError:
                    pass  # deleted: dropped
            carry["read_files"] = fresh_reads

        for field in _mirrored_fields():
            prior = carry.get(field)
            current = execution_context.get(field)
            if isinstance(current, set) and isinstance(prior, set):
                # sorted(): carried entries seed a RecencySet's order, and a set iterates
                # in a per-process hash order; a stable order keeps the pin byte-stable.
                current.update(sorted(prior))
            elif isinstance(current, dict) and isinstance(prior, dict):
                # A skill already in the conversation is not pulled again.
                current.update(prior)
            elif isinstance(current, bool) and prior:
                setattr(execution_context.home(field), field, True)

    def update(self, execution_context: ExecutionContext) -> None:
        """Fold what this query learnt into the carry."""
        carry = self.fields
        for field in _mirrored_fields():
            current = execution_context.get(field)
            prior = carry.get(field)
            if isinstance(current, set):
                carry[field] = (prior if isinstance(prior, set) else set()) | current
            elif isinstance(current, dict):
                # A reservation that never resolved is not a load: carrying "pending"
                # would bar the skill for the rest of the session.
                merged = dict(prior) if isinstance(prior, dict) else {}
                merged.update({k: v for k, v in current.items() if v != "pending"})
                carry[field] = merged
            elif isinstance(current, bool) and current:
                carry[field] = True
        carry["last_query_written_files"] = set(execution_context.edits.dirty_written_files)
        # A file written this query was read before the write: that read is stale.
        dirty: set[str] = execution_context.edits.dirty_written_files
        read_mtimes: dict[str, float] = carry.setdefault("_read_mtimes", {})
        if dirty:
            carry_reads: set[str] = carry.get("read_files", set())
            carry_reads -= dirty
            for p in dirty:
                read_mtimes.pop(p, None)
        # Every carried read is stamped with the mtime seen now: the next query compares
        # against the file as this query left it.
        carried_reads: set[str] = carry.get("read_files", set())
        for p in carried_reads:
            try:
                read_mtimes[p] = os.path.getmtime(absolute_workspace_path(p))
            except OSError:
                read_mtimes.pop(p, None)
        for p in [p for p in read_mtimes if p not in carried_reads]:
            del read_mtimes[p]

    def discard_path(self, path: str) -> None:
        """Forget a deleted file in every carried field that holds file paths.

        ``inspected_dirs`` holds directories and keeps the one the file was in.
        """
        for field in carry_path_fields():
            self.fields.get(field, set()).discard(path)
        self.fields.get("_read_mtimes", {}).pop(path, None)
