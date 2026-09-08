"""Atomic snapshots of the file set under optimization.

WHY A SET AND NOT A FILE. The ratchet used to track one ``proxy_source_path`` and
snapshot it with ``shutil.copy2``. Generalising that to several files by keeping a best
*per file* would be wrong in a way that does not announce itself: restoring "the best
version of each" can assemble a combination that was never measured together — file A
from run 7 beside file B from run 12, after a signature changed in run 9. It may even
run, and produce a number that means nothing. The unit of acceptance has to be the tree
state that was measured, taken and restored as one.

WHY GIT. "Snapshot a set of paths, label it, restore it atomically, diff two of them" is
version control. Writing a bespoke one means maintaining a format that can be corrupted
in ways git's cannot. So this drives a *shadow* repository: the git dir lives in the proxy
store, the work tree is the workspace, and only the declared paths are ever added. The
user's own ``.git`` is never opened, no branch or index of theirs is touched, and the
workspace does not need to be a repository at all — the one this was built against is not.

Every invocation carries its own identity and disables signing: a machine with no
``user.email`` configured, or with a global signing hook, must not turn "record the state
I just measured" into a failed run. ``add -f`` for the same reason — a global
``core.excludesFile`` matching ``*.py`` is somebody's real configuration, and it is not
this store's business to honour it.

FALLBACK. Where git is unavailable the same API copies the tracked paths into a directory
per snapshot. Same semantics including atomicity, more disk, no diff. A fallback, not a
design: it exists so an unusual machine degrades instead of failing.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_shared"))
import proc_run  # noqa: E402  (kills the whole process group on timeout)

# Long enough for a large tree on a slow filesystem, short enough that a hung git can
# never become a hung optimisation run.
_GIT_TIMEOUT_S = 120

_IDENTITY = (
    "-c", "user.name=MIMIR proxy ratchet",
    "-c", "user.email=ratchet@mimir.invalid",
    "-c", "commit.gpgsign=false",
    "-c", "gc.auto=0",
)


def git_available() -> bool:
    try:
        out = proc_run.run(["git", "--version"], timeout=10,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return out.returncode == 0
    except Exception:
        return False


def _git(git_dir: str, work_tree: str, *args: str) -> subprocess.CompletedProcess:
    argv = ["git", "--git-dir", git_dir, "--work-tree", work_tree, *_IDENTITY, *args]
    return proc_run.run(argv, timeout=_GIT_TIMEOUT_S, cwd=work_tree,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _rel(work_tree: str, paths: list[str]) -> list[str]:
    """Tracked paths as work-tree-relative pathspecs, which is what git wants."""
    out = []
    for p in paths:
        rp = os.path.relpath(os.path.abspath(p), os.path.abspath(work_tree))
        out.append(rp)
    return out


# ── git-backed implementation ────────────────────────────────────────────────

def _git_snapshot(git_dir: str, work_tree: str, paths: list[str], message: str) -> str | None:
    if not os.path.isdir(git_dir):
        r = proc_run.run(["git", "init", "--bare", "--quiet", git_dir], timeout=_GIT_TIMEOUT_S,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            return None
    rels = _rel(work_tree, paths)
    if _git(git_dir, work_tree, "add", "-f", "--", *rels).returncode != 0:
        return None
    # --allow-empty: re-measuring an unchanged tree is a legitimate run, and it must
    # still get an identity of its own in the ledger.
    c = _git(git_dir, work_tree, "commit", "--allow-empty", "-q", "-m", message)
    if c.returncode != 0:
        return None
    h = _git(git_dir, work_tree, "rev-parse", "HEAD")
    return h.stdout.strip() if h.returncode == 0 else None


def _git_restore(git_dir: str, work_tree: str, paths: list[str], snapshot_id: str) -> bool:
    rels = _rel(work_tree, paths)
    return _git(git_dir, work_tree, "checkout", snapshot_id, "--", *rels).returncode == 0


def _git_diff(git_dir: str, work_tree: str, a: str, b: str) -> str:
    r = _git(git_dir, work_tree, "diff", "--stat", a, b)
    return r.stdout if r.returncode == 0 else ""


# ── copy-backed fallback ─────────────────────────────────────────────────────

def _copy_root(git_dir: str) -> str:
    """Snapshots live beside the git dir so one `clean` removes both."""
    return os.path.join(os.path.dirname(git_dir), "tree_snapshots")


def _copy_snapshot(git_dir: str, work_tree: str, paths: list[str], message: str) -> str | None:
    snap_id = uuid.uuid4().hex[:12]
    dest_root = os.path.join(_copy_root(git_dir), snap_id)
    try:
        for rel in _rel(work_tree, paths):
            src = os.path.join(work_tree, rel)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        os.makedirs(dest_root, exist_ok=True)
        with open(os.path.join(dest_root, ".message"), "w", encoding="utf-8") as fh:
            fh.write(message)
    except OSError:
        return None
    return snap_id


def _copy_restore(git_dir: str, work_tree: str, paths: list[str], snapshot_id: str) -> bool:
    src_root = os.path.join(_copy_root(git_dir), snapshot_id)
    if not os.path.isdir(src_root):
        return False
    # Two passes: stage every file beside its target first, then rename them all. A
    # restore that fails halfway would leave a tree that was never measured — the exact
    # thing this module exists to prevent.
    staged: list[tuple[str, str]] = []
    try:
        for rel in _rel(work_tree, paths):
            src = os.path.join(src_root, rel)
            if not os.path.isfile(src):
                continue
            final = os.path.join(work_tree, rel)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            tmp = final + ".mimir_restore_tmp"
            shutil.copy2(src, tmp)
            staged.append((tmp, final))
        for tmp, final in staged:
            os.replace(tmp, final)
    except OSError:
        for tmp, _ in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False
    return True


# ── public API ───────────────────────────────────────────────────────────────

def snapshot(git_dir: str, work_tree: str, paths: list[str], message: str) -> str | None:
    """Record the current state of *paths*. Returns an opaque id, or None on failure."""
    if not paths:
        return None
    if git_available():
        sid = _git_snapshot(git_dir, work_tree, paths, message)
        if sid:
            return sid
    return _copy_snapshot(git_dir, work_tree, paths, message)


def restore(git_dir: str, work_tree: str, paths: list[str], snapshot_id: str) -> bool:
    """Put *paths* back to *snapshot_id*, all of them or none."""
    if not snapshot_id or not paths:
        return False
    # A copy-backed id is a short hex tag with a directory to match; anything else is a
    # git object. Checked by existence rather than by shape, so the two never disagree.
    if os.path.isdir(os.path.join(_copy_root(git_dir), snapshot_id)):
        return _copy_restore(git_dir, work_tree, paths, snapshot_id)
    return _git_restore(git_dir, work_tree, paths, snapshot_id)


def diff(git_dir: str, work_tree: str, a: str, b: str) -> str:
    """A stat diff between two snapshots; empty when unavailable (copy fallback)."""
    if not (a and b) or not git_available():
        return ""
    return _git_diff(git_dir, work_tree, a, b)


def fingerprint(work_tree: str, paths: list[str]) -> str:
    """Content hash of the tracked paths, independent of the snapshot backend.

    Used to answer "is the tree still what this run measured?" without asking git — a
    question the ledger needs even when the snapshot backend is the fallback.
    """
    h = hashlib.sha256()
    for rel in sorted(_rel(work_tree, paths)):
        h.update(rel.encode("utf-8"))
        try:
            with open(os.path.join(work_tree, rel), "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]
