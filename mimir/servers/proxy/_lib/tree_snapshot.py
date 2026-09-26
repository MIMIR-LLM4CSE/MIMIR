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

WHAT A RESTORE WRITES. Only the files whose content actually differs from the snapshot.
That is not an optimisation of disk writes — a restored file is deliberately stamped with
the current time, because a build system decides what to recompile by comparing mtimes,
and a file dated before the artifact built from it would be skipped. A fresh mtime is
therefore an *instruction to recompile*, and giving it to a file whose bytes never
changed is a false one. Writing the whole tracked set, as both backends used to, made
every reset_to_best after a one-file edit cost a full rebuild. Both backends now decide
by content, never by stat: git by blob sha via ls-tree/hash-object, the fallback by
reading both files. Not ``filecmp`` — its cache is keyed on (size, mtime), so an edit
landing in the same filesystem tick as a restore would come back "identical".
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


def _snapshot_blobs(git_dir: str, work_tree: str, snapshot_id: str,
                    rels: list[str]) -> dict[str, str] | None:
    """Each tracked path's blob sha *in the snapshot*, or None if git would not say.

    ``ls-tree`` reads the commit and nothing else, so no index is consulted and a
    stale one in the shadow repository cannot colour the answer.
    """
    r = _git(git_dir, work_tree, "ls-tree", "-r", "-z", snapshot_id, "--", *rels)
    if r.returncode != 0:
        return None
    out: dict[str, str] = {}
    for record in r.stdout.split("\0"):
        if not record:
            continue
        # "<mode> <type> <sha>\t<path>"
        meta, _, path = record.partition("\t")
        fields = meta.split()
        if len(fields) == 3 and path:
            out[path] = fields[2]
    return out


def _worktree_blobs(git_dir: str, work_tree: str,
                    rels: list[str]) -> dict[str, str] | None:
    """Each existing path's blob sha *on disk*, hashed from its bytes.

    ``hash-object`` is a pure function of the file's content — it touches neither the
    index nor the object store — so comparing these to the snapshot's is a content
    comparison and never a stat comparison.
    """
    present = [rel for rel in rels if os.path.isfile(os.path.join(work_tree, rel))]
    if not present:
        return {}
    r = _git(git_dir, work_tree, "hash-object", "--", *present)
    if r.returncode != 0:
        return None
    shas = r.stdout.split()
    if len(shas) != len(present):
        return None
    return dict(zip(present, shas))


def _git_restore(git_dir: str, work_tree: str, paths: list[str],
                 snapshot_id: str) -> list[str] | None:
    """Check out only the tracked paths whose content actually differs.

    ``git checkout <commit> -- <pathspec>`` rewrites every path it is handed, whether
    or not the content changed, and a rewritten file carries the current time. Handing
    it the whole tracked set — which is what this did — told the build system that
    every one of those files was new, so a reset_to_best undoing a one-file edit cost a
    rebuild of all of them, and of everything downstream of a header among them.

    So the set is narrowed first, by content. A path that is byte-identical to the
    snapshot needs no write at all: whatever artifact exists was built from exactly
    those bytes, and leaving its mtime alone is the correct answer rather than a
    shortcut. Anything missing from disk is restored, since "identical" is meaningless
    there.

    Either probe failing falls back to the unconditional checkout: degrading towards
    "rebuilt more than necessary" is acceptable, degrading towards "did not restore"
    is not.
    """
    rels = _rel(work_tree, paths)
    snapshot = _snapshot_blobs(git_dir, work_tree, snapshot_id, rels)
    current = _worktree_blobs(git_dir, work_tree, rels) if snapshot is not None else None
    if snapshot is None or current is None:
        if _git(git_dir, work_tree, "checkout", snapshot_id, "--", *rels).returncode != 0:
            return None
        return list(rels)

    stale = [rel for rel, sha in snapshot.items() if current.get(rel) != sha]
    if not stale:
        return []
    if _git(git_dir, work_tree, "checkout", snapshot_id, "--", *stale).returncode != 0:
        return None
    return stale


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


def _same_bytes(p1: str, p2: str) -> bool:
    """Whether two files hold the same bytes. False if either cannot be read."""
    try:
        if os.path.getsize(p1) != os.path.getsize(p2):
            return False
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            while True:
                b1, b2 = f1.read(65536), f2.read(65536)
                if b1 != b2:
                    return False
                if not b1:
                    return True
    except OSError:
        return False


def _copy_restore(git_dir: str, work_tree: str, paths: list[str],
                  snapshot_id: str) -> list[str] | None:
    src_root = os.path.join(_copy_root(git_dir), snapshot_id)
    if not os.path.isdir(src_root):
        return None
    # Two passes: stage every file beside its target first, then rename them all. A
    # restore that fails halfway would leave a tree that was never measured — the exact
    # thing this module exists to prevent.
    staged: list[tuple[str, str, str]] = []
    try:
        for rel in _rel(work_tree, paths):
            src = os.path.join(src_root, rel)
            if not os.path.isfile(src):
                continue
            final = os.path.join(work_tree, rel)
            # Compared by reading both files, never by stat and never through
            # ``filecmp``: its module-level cache is keyed on (size, mtime), and an edit
            # landing in the same filesystem tick as a restore leaves that key unchanged
            # while the content differs — so a cached "identical" would skip a file that
            # genuinely needs restoring. Editing immediately after reset_to_best is the
            # normal loop, not a corner case.
            if _same_bytes(src, final):
                continue
            os.makedirs(os.path.dirname(final), exist_ok=True)
            tmp = final + ".mimir_restore_tmp"
            shutil.copy2(src, tmp)
            staged.append((tmp, final, rel))
        for tmp, final, _rel_name in staged:
            os.replace(tmp, final)
            # copy2 carried the snapshot's mtime across, which would date a
            # restored file to before the artifact built from it. `make` reads
            # exactly that comparison and would skip the rebuild, so the next
            # run would measure the binary of the attempt this restore is
            # undoing. Stamping the file as new is what lets the build system
            # decide correctly.
            #
            # Which is precisely why only the files above reach this loop. A
            # fresh mtime is an instruction to recompile, and handing it to a
            # file whose bytes never changed is a false one: the artifact beside
            # it was built from exactly these bytes. Stamping the whole tracked
            # set — which is what this did — turned every reset_to_best after a
            # one-file edit into a full rebuild, and into a rebuild of
            # everything downstream when a header was among them.
            os.utime(final, None)
    except OSError:
        for tmp, _final, _rel_name in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return None
    return [rel for _tmp, _final, rel in staged]


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


def restore(git_dir: str, work_tree: str, paths: list[str],
            snapshot_id: str) -> list[str] | None:
    """Put *paths* back to *snapshot_id*, all of them or none.

    Returns the work-tree-relative paths it actually had to write, or None on failure.
    **An empty list is a success**, not a failure: it means the tree was already in that
    state and nothing needed touching — so ``if not restore(...)`` is the one way to
    read this wrong, and callers test ``is None``.

    Only files whose content differs are written, because writing one is how this tells
    the build system to recompile it (see ``_git_restore`` and ``_copy_restore``). The
    end state is the snapshot either way; what changes is how much of the project has to
    be rebuilt to measure it.
    """
    if not snapshot_id or not paths:
        return None
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
