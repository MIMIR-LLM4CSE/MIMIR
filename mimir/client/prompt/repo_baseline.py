from __future__ import annotations

import os
from typing import Any

from ..config.constants import BASELINE_SKIP_DIRS
from ..context.execution_context import BASELINE_SEEDED_DIRS


def canonical_query_key(query: str) -> str:
    return " ".join((query or "").strip().lower().split())[:160]


def _workspace_root() -> str:
    """The directory the baseline walks: the same root the client resolves paths against."""
    return os.environ.get("SEARCH_ROOT") or os.getcwd()


def build_repo_baseline_snapshot(
    *,
    root: str | None = None,
    max_depth: int = 2,
    max_entries: int = 150,
    max_chars: int = 15000,
) -> dict[str, Any]:
    """Build a one-shot structural snapshot of the workspace, in memory.

    Self-contained: walks the filesystem directly with its own ignore set, so it
    works whether or not the search server is registered. The result is foundational
    agent-identity context (held for the session, never written to disk); the search
    tools remain available for the model's deeper, query-specific exploration. Returns
    a plain dict (``context`` / ``inspected_dirs`` / ``searched``) read by the system-
    prompt builder and execution-context seeding.
    """
    root = root or _workspace_root()
    # The root line is the ABSOLUTE path, not the basename. Rendered bare ("codes/") the
    # root is indistinguishable from a subdirectory, so a model told to write "outside
    # the codes directory" reads it as a child of the workspace and writes straight into
    # the directory it was avoiding — while sincerely reporting the constraint met.
    root_name = os.path.abspath(root.rstrip(os.sep)) or "."

    lines: list[str] = []
    top_dirs: list[str] = []
    top_files: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue

        # Prune noise in-place so os.walk does not descend into it.
        dirnames[:] = sorted(d for d in dirnames if d not in BASELINE_SKIP_DIRS)
        filenames = sorted(filenames)
        if rel == ".":
            top_dirs = list(dirnames)
            top_files = list(filenames)

        indent = "  " * depth
        label = root_name if rel == "." else os.path.basename(dirpath)
        lines.append(f"{indent}{label}/ ({len(dirnames)} dirs, {len(filenames)} files)")
        if len(lines) >= max_entries:
            truncated = True
            break

        for fn in filenames:
            try:
                size_kb = round(os.path.getsize(os.path.join(dirpath, fn)) / 1024, 1)
                lines.append(f"{indent}  {fn} [{size_kb} KB]")
            except OSError:
                lines.append(f"{indent}  {fn}")
            if len(lines) >= max_entries:
                truncated = True
                break
        if truncated:
            break

    sections: list[str] = []
    if lines:
        # State the root absolutely. The tree renders from the basename down, which
        # reads as if the workspace were the whole world — with the absolute path stated
        # nowhere else, a destination outside it is not discouraged but unnameable.
        sections.append(f"Workspace root (absolute): {root}")
        sections.append("Structure summary:\n" + "\n".join(lines))
    if top_dirs:
        sections.append("Top-level directories: " + ", ".join(top_dirs[:20]))
    if top_files:
        sections.append("Top-level files: " + ", ".join(top_files[:20]))

    context_text = "\n\n".join(sections)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    return {
        # "." is the only dir seeded as inspected — orientation that cannot green-light
        # a write on its own. The discovery-evidence counter discounts exactly these
        # entries, so a snapshot can never pre-satisfy a gate.
        "context": context_text,
        "inspected_dirs": sorted(BASELINE_SEEDED_DIRS) if context_text else [],
        "searched": bool(context_text),
        "root": root,
    }


def seed_execution_context_from_baseline(
    *,
    execution_context: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> None:
    """Seed low-risk orientation from the in-memory baseline.

    Only ``inspected_dirs`` is seeded — it cannot green-light a write/delete on its own
    (those gates still require exact-path/existence evidence). We deliberately do NOT seed
    ``searched=True``: that global flag would mark repo discovery "done" for the whole
    policy from a snapshot, suppressing the agent's first real search. Let the agent's own
    search set it.
    """
    if not baseline:
        return
    for d in baseline.get("inspected_dirs", []):
        if isinstance(d, str) and d:
            execution_context["inspected_dirs"].add(d)
