"""What a bash command that was not read-only changed on disk, reported to the model.

Observed, never parsed from the command: in a git repo, the delta of ``git status`` and
``git diff --numstat`` around the call; elsewhere, a bounded scan of the directories it
names. Best-effort: any failure costs the annotation, never the call.
"""

from __future__ import annotations

import difflib
import logging
import os
import subprocess
from dataclasses import dataclass, field

from ..config.limits import (
    BASH_EFFECT_DIFF_MAX_LINES,
    BASH_EFFECT_DUP_MAX_LINES,
    BASH_EFFECT_DUP_MIN_BLOCK,
    BASH_EFFECT_MAX_FILES,
    BASH_EFFECT_SNAPSHOT_MAX_BYTES,
    BASH_EFFECT_SNAPSHOT_MAX_FILES,
)
from ..tools.bash_classify import bash_command_is_readonly, classify_bash_command
from ..tools.capabilities import shell_command_of
from ..context.execution_context import ExecutionContext
from ..context.signals import SOURCE_FILE_EXTENSIONS
from ..core.ports import ExecutorAgent

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECS = 10


def _git(root: str, *args: str) -> str | None:
    """Run a git command in *root*, or None when git cannot answer."""
    try:
        proc = subprocess.run(
            ("git", *args), cwd=root, capture_output=True, text=True,
            timeout=_GIT_TIMEOUT_SECS,
            # git asks for a passphrase or an editor on stdin given the chance, and this
            # runs mid-turn: with nothing to read it fails instead of holding the turn.
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        logger.debug("_git: a best-effort step failed and was skipped", exc_info=True)
        return None
    return proc.stdout if proc.returncode == 0 else None


def _git_state(root: str) -> tuple[dict[str, str], dict[str, tuple[int, int]]] | None:
    """(porcelain status by path, numstat counts by path), or None outside a repo."""
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status is None:
        return None
    by_path: dict[str, str] = {}
    for line in status.splitlines():
        if len(line) > 3:
            by_path[line[3:].strip().strip('"')] = line[:2]
    counts: dict[str, tuple[int, int]] = {}
    numstat = _git(root, "diff", "--numstat") or ""
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            counts[parts[2].strip()] = (int(parts[0]), int(parts[1]))
    return by_path, counts


def _read_lines(path: str) -> list[str] | None:
    """Text lines of *path*, or None when it is missing, binary or oversized."""
    try:
        if os.path.getsize(path) > BASH_EFFECT_SNAPSHOT_MAX_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="strict") as handle:
            return handle.read().splitlines()
    except Exception:
        logger.debug("_read_lines: a best-effort step failed and was skipped", exc_info=True)
        return None


def _is_source(path: str) -> bool:
    return os.path.splitext(path)[1] in SOURCE_FILE_EXTENSIONS


def repeated_block(added: list[str]) -> tuple[int, int] | None:
    """``(block length, repeats)`` when *added* is one block repeated end to end, else None.

    A period, not a repeated window: the signature of an unanchored address matching
    everywhere.
    """
    lines = [ln for ln in added if ln.strip()]
    if len(lines) < 2 * BASH_EFFECT_DUP_MIN_BLOCK or len(lines) > BASH_EFFECT_DUP_MAX_LINES:
        return None
    # From the shortest period up, so the block reported is the real repeating unit. A
    # sequence with period 2 also has period 4, 6, 8 …; starting higher would describe
    # "a, b" five times as "a, b, a, b" twice, which is true and useless. The floor is
    # on the total instead: six repeating lines is where a repetition stops being the
    # incidental recurrence every source file has.
    for period in range(1, len(lines) // 2 + 1):
        if all(lines[i] == lines[i + period] for i in range(len(lines) - period)):
            return period, len(lines) // period
    return None


@dataclass
class _Probe:
    """Pre-command state, held across the dispatch so the delta can be read after."""

    root: str
    git: tuple[dict[str, str], dict[str, tuple[int, int]]] | None = None
    snapshots: dict[str, list[str] | None] = field(default_factory=dict)


def _candidate_dirs(
    agent: ExecutorAgent, command: str, root: str, execution_context: ExecutionContext | None,
) -> list[str]:
    """Directories to watch when git cannot answer: the root, those written to this query, and
    the command's own operands.
    """
    dirs: set[str] = {root}
    dirty = execution_context.edits.dirty_written_files if execution_context is not None else ()
    for path in dirty:
        dirs.add(os.path.dirname(str(path)))
    for segment in classify_bash_command(command) or []:
        for operand in segment.operands:
            try:
                full = agent.normalize_workspace_path(operand) or operand
            except Exception:
                full = operand
            if not os.path.isabs(full):
                full = os.path.join(root, full)
            dirs.add(full if os.path.isdir(full) else os.path.dirname(full))
    return [d for d in dirs if d and os.path.isdir(d)][:BASH_EFFECT_SNAPSHOT_MAX_FILES]


def _scan(dirs: list[str]) -> dict[str, list[str] | None]:
    """Content of the source files directly inside *dirs*, capped in count."""
    out: dict[str, list[str] | None] = {}
    for directory in dirs:
        try:
            entries = list(os.scandir(directory))
        except Exception:
            logger.debug("_scan: a best-effort step failed and was skipped", exc_info=True)
            continue
        for entry in entries:
            if len(out) >= BASH_EFFECT_SNAPSHOT_MAX_FILES:
                return out
            if entry.is_file() and _is_source(entry.path):
                out[entry.path] = _read_lines(entry.path)
    return out


def capture(
    agent: ExecutorAgent, tool_name: str, arguments: dict,
    execution_context: ExecutionContext | None = None,
) -> _Probe | None:
    """Snapshot enough state to report what this call changes, or None to skip it."""
    try:
        command = shell_command_of(tool_name, arguments, agent.tool_caps)
        if not command or bash_command_is_readonly(command):
            return None
        root = agent.workspace_root or os.getcwd()
        probe = _Probe(root=root)
        probe.git = _git_state(root)
        if probe.git is None:
            probe.snapshots = _scan(_candidate_dirs(agent, command, root, execution_context))
        return probe
    except Exception:
        logger.debug("capture: a best-effort step failed and was skipped", exc_info=True)
        return None


def _git_changes(probe: _Probe) -> list[tuple[str, int, int, bool]]:
    """(relpath, added, removed, was_clean) for each file this command touched."""
    after = _git_state(probe.root)
    if after is None or probe.git is None:
        return []
    before_status, before_counts = probe.git
    after_status, after_counts = after
    changed: list[tuple[str, int, int, bool]] = []
    for path in sorted(set(after_status) | set(after_counts)):
        was_clean = path not in before_status and path not in before_counts
        if after_status.get(path) == before_status.get(path) and \
                after_counts.get(path) == before_counts.get(path):
            continue
        added, removed = after_counts.get(path, (0, 0))
        old_added, old_removed = before_counts.get(path, (0, 0))
        if path not in before_counts and path not in after_counts:
            # Untracked: git reports no counts, so the whole file is the addition.
            lines = _read_lines(os.path.join(probe.root, path))
            added, removed = (len(lines) if lines is not None else 0), 0
        else:
            added, removed = added - old_added, removed - old_removed
        changed.append((path, added, removed, was_clean))
    return changed


def _added_lines(probe: _Probe, relpath: str, was_clean: bool) -> list[str]:
    """Lines this command added to *relpath*, claimed only for a file that was clean before."""
    if not was_clean:
        return []
    full = os.path.join(probe.root, relpath)
    diff = _git(probe.root, "diff", "-U0", "--", relpath)
    if diff is None or not diff.strip():
        lines = _read_lines(full)
        return lines or []
    return [ln[1:] for ln in diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")]


def _clip(lines: list[str], budget: int) -> list[str]:
    """*lines* trimmed to *budget*, saying how many it dropped."""
    if len(lines) <= budget:
        return lines
    return lines[:budget] + [f"  … {len(lines) - budget} more diff lines"]


def _git_diff_text(probe: _Probe, relpath: str, budget: int) -> list[str]:
    """The command's diff for *relpath*, as the model would have got from an edit tool."""
    diff = _git(probe.root, "diff", "-U1", "--", relpath)
    if diff:
        body = [ln for ln in diff.splitlines()
                if not ln.startswith(("diff --git", "index ", "--- ", "+++ "))]
        return _clip(body, budget)
    lines = _read_lines(os.path.join(probe.root, relpath)) or []
    return _clip([f"+{ln}" for ln in lines], budget)


def _fallback_changes(probe: _Probe) -> list[tuple[str, int, int, list[str]]]:
    """(path, added, removed, added lines) from the no-git content snapshots."""
    changed: list[tuple[str, int, int, list[str]]] = []
    for path, before in probe.snapshots.items():
        after = _read_lines(path)
        if after == before:
            continue
        added_lines = [
            line[2:] for line in difflib.ndiff(before or [], after or [])
            if line.startswith("+ ")
        ]
        removed = sum(
            1 for line in difflib.ndiff(before or [], after or []) if line.startswith("- ")
        )
        changed.append((path, len(added_lines), removed, added_lines))
    return changed


def report(probe: _Probe | None, root_display: str = "") -> str:
    """The BASH_EFFECT annotation for a finished call, or "" when nothing changed."""
    if probe is None:
        return ""
    try:
        rows: list[str] = []
        warnings: list[str] = []
        # One diff budget for the whole report: eight files each showing forty lines is
        # not a diff to check, it is the file back again.
        budget = BASH_EFFECT_DIFF_MAX_LINES
        if probe.git is not None:
            entries = _git_changes(probe)
            for relpath, added, removed, was_clean in entries[:BASH_EFFECT_MAX_FILES]:
                rows.append(f"- {relpath} — +{max(added, 0)}/-{max(removed, 0)} lines")
                diff = _git_diff_text(probe, relpath, budget) if budget > 0 else []
                budget -= len(diff)
                rows.extend(f"  {line}" for line in diff)
                found = repeated_block(_added_lines(probe, relpath, was_clean))
                if found:
                    warnings.append(_duplication_line(relpath, *found))
            total = len(entries)
        else:
            fallback = _fallback_changes(probe)
            for path, added, removed, added_lines in fallback[:BASH_EFFECT_MAX_FILES]:
                shown = os.path.relpath(path, probe.root) if probe.root else path
                rows.append(f"- {shown} — +{added}/-{removed} lines")
                diff = _clip(
                    [ln for ln in difflib.unified_diff(
                        probe.snapshots.get(path) or [], _read_lines(path) or [],
                        lineterm="", n=1)
                     if not ln.startswith(("---", "+++"))],
                    budget,
                ) if budget > 0 else []
                budget -= len(diff)
                rows.extend(f"  {line}" for line in diff)
                found = repeated_block(added_lines)
                if found:
                    warnings.append(_duplication_line(shown, *found))
            total = len(fallback)
        if not rows:
            return ""
        shown_files = min(total, BASH_EFFECT_MAX_FILES)
        more = f"\n- … and {total - shown_files} more" if total > shown_files else ""
        text = (
            "\n\nBASH_EFFECT: this command changed files on disk. A shell write returns "
            "no diff, so this is the only report of it:\n" + "\n".join(rows) + more
        )
        if warnings:
            text += "\n" + "\n".join(warnings)
        return text
    except Exception:
        logger.debug("report: a best-effort step failed and was skipped", exc_info=True)
        return ""


def _duplication_line(path: str, block: int, times: int) -> str:
    return (
        f"DUPLICATION_SUSPECTED: in {path}, everything added is the same {block}-line "
        f"block repeated {times} times. An address or pattern that is not unique applies "
        f"at every place it matches — read the file back and check the scope before "
        f"building on this."
    )


def created_paths(probe: _Probe | None) -> list[str]:
    """Absolute paths this command created, for the fork/placement annotations."""
    if probe is None or probe.git is None:
        return []
    try:
        after = _git_state(probe.root)
        if after is None:
            return []
        before_status, _ = probe.git
        return [
            os.path.join(probe.root, path)
            for path, status in sorted(after[0].items())
            if status.strip() == "??" and path not in before_status
        ]
    except Exception:
        logger.debug("created_paths: a best-effort step failed and was skipped", exc_info=True)
        return []
