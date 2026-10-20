"""A copy of the repository for a sub-agent's own axis: a git worktree on a branch of
its own, committed and handed back when the child ends, and swept when abandoned."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from ....shared.atomic import write_json

# On the local disk, not in the state dir: a home under quota cannot hold a checkout
# plus everything a build writes. Deliberately NOT under the scratchpad either — that
# path is writable without approval (state_paths.standing_roots), and a copy placed
# there would take every edit the child makes out of the approval layer.
WORKTREE_BASE = os.path.join(
    os.path.abspath(os.environ.get("TMPDIR") or "/tmp"),
    f"mimir-worktrees-{os.getuid()}",
)


GIT_TIMEOUT_SECS = 120


# How many *finished* copies one session keeps, the copy being made not counted. A
# finished copy is worth keeping — it holds the build tree, which git never had and
# which cost the axis its compile — but nothing reclaims /tmp on its own, so this count
# is the bound: making the next copy drops the oldest beyond it.
SUBAGENT_WORKTREES_KEPT = 3


def git(args: list[str], cwd: str) -> tuple[bool, str]:
    """Run one git command. Returns (ok, output) and never raises."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=GIT_TIMEOUT_SECS,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    out = (done.stdout or "") + (done.stderr or "")
    return done.returncode == 0, out.strip()


def repo_root() -> str:
    """The git repository this agent works in, or "" when there is none."""
    root = os.environ.get("MCP_FILES_ROOT") or os.getcwd()
    ok, out = git(["rev-parse", "--show-toplevel"], root)
    return out if ok and out else ""


def sibling_cards(session: str) -> tuple[str, list[dict]]:
    """``(parent dir, cards)`` of every child of this session's parent, newest first; best-effort."""
    parent = session.rsplit("/subagents/", 1)[0] if "/subagents/" in session else ""
    if not parent:
        return "", []
    try:
        from ....shared.state_paths import subagents_dir
        base = subagents_dir(parent)
        names = sorted(os.listdir(base))
    except Exception:
        return "", []
    # Newest first, by the card's own clock and then by when it was last written. The
    # timestamp has second resolution and a fan-out starts its children inside one
    # second, so on its own it would leave the order of a tie to the filesystem.
    rows = []
    for name in names:
        card_path = os.path.join(base, name, "subagent.json")
        try:
            with open(card_path, encoding="utf-8") as fh:
                card = json.load(fh)
            rows.append((card.get("started_at", ""), os.path.getmtime(card_path), card))
        except Exception:
            continue
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return base, [row[2] for row in rows]


def card_is_live(card: dict) -> bool:
    """Whether a "running" card's process still exists (no pid, or another user's, counts as live)."""
    if card.get("state") != "running":
        return False
    pid = card.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def commit_copy(path: str, message: str) -> tuple[list[str], str]:
    """Stage and commit whatever the copy holds; ``(files, diffstat)``.

    Called when the child finishes and before a copy is reclaimed, so no work is dropped.
    """
    git(["add", "-A"], path)
    ok, changed = git(["diff", "--cached", "--name-only"], path)
    files = [f for f in changed.splitlines() if f] if ok else []
    ok, stat = git(["diff", "--cached", "--stat"], path)
    diffstat = stat if ok else ""
    if files:
        git(["-c", "user.name=MIMIR", "-c", "user.email=mimir@localhost",
              "commit", "-m", message], path)
    return files, diffstat


def drop_branch_if_empty(repo: str, branch: str, base: str) -> bool:
    """Delete *branch* if it is still at the *base* commit it was cut from; whether it went."""
    if not branch or not base:
        return False
    ok, tip = git(["rev-parse", branch], repo)
    if not ok or tip != base:
        return False
    deleted, _ = git(["branch", "-D", branch], repo)
    return deleted


def marker_path(repo: str, child_key: str) -> str:
    """Where a copy's marker lives: beside the copy, never inside it.

    Inside, ``git add -A`` would commit it into the axis. Beside, it survives the copy
    being emptied and stays out of every diff.
    """
    return os.path.join(WORKTREE_BASE, os.path.basename(repo), f"{child_key}.mimir.json")


def write_marker(path: str, marker: dict) -> None:
    """Record who a copy belongs to, before it can be used. Best effort."""
    try:
        write_json(path, marker)
    except OSError:
        pass


def drop_copy(wt: dict, child_key: str = "") -> bool:
    """Commit whatever the copy holds, remove it, and drop a branch that carries nothing.

    The single way a copy ever goes, so the commit can never be skipped by one caller
    and not the other: a copy taken with uncommitted work in it is work destroyed.
    """
    path, repo = wt.get("path", ""), wt.get("repo", "")
    if not path or not repo:
        return False
    commit_copy(path, f"sub-agent {os.path.basename(path)}: reclaimed unfinished work")
    removed, _ = git(["worktree", "remove", "--force", path], repo)
    git(["worktree", "prune"], repo)
    if not removed:
        return False
    drop_branch_if_empty(repo, wt.get("branch") or "", wt.get("base") or "")
    with contextlib.suppress(OSError):
        os.remove(marker_path(repo, child_key or os.path.basename(path)))
    return True


def sweep_orphan_copies(repo: str) -> None:
    """Drop copies of this repository whose owner process and card are both gone.

    A copy without a marker is judged by git: one it does not register is unused.
    """
    base = os.path.join(WORKTREE_BASE, os.path.basename(repo))
    if not os.path.isdir(base):
        return
    ok, listed = git(["worktree", "list", "--porcelain"], repo)
    registered = {line.split(" ", 1)[1] for line in listed.splitlines()
                  if ok and line.startswith("worktree ")}
    try:
        from ....shared.state_paths import sessions_root
        sessions = sessions_root()
    except Exception:
        return
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        marker = marker_path(repo, name)
        try:
            with open(marker, encoding="utf-8") as fh:
                owner = json.load(fh)
        except (OSError, json.JSONDecodeError):
            if os.path.realpath(path) not in {os.path.realpath(p) for p in registered}:
                shutil.rmtree(path, ignore_errors=True)
            continue
        if card_is_live({"state": "running", "pid": owner.get("pid")}):
            continue
        if os.path.isdir(os.path.join(sessions, owner.get("session", ""))):
            continue          # its session still holds its card; the per-session rule owns it
        drop_copy(owner, name)


def reclaim_worktrees(session: str, keep: int) -> None:
    """Drop this session's finished copies on disk beyond the *keep* most recent.

    Running children (by the pid on their card) are never counted or removed.
    """
    _, cards = sibling_cards(session)
    copies = [w for w in (c.get("workspace") or {} for c in cards if not card_is_live(c))
              if w.get("path") and w.get("repo") and os.path.isdir(w["path"])]
    for wt in copies[keep:]:
        drop_copy(wt)


def create_worktree(child_key: str, session: str = "") -> tuple[dict | None, str]:
    """Add a worktree for this child. Returns (descriptor, error message)."""
    repo = repo_root()
    if not repo:
        return None, (
            "a sub-agent that writes needs a copy of the repository, and this workspace "
            "is not a git repository — so there is none to make. Delegate the reading "
            "and do the writing yourself."
        )
    branch = f"mimir/{child_key}"
    path = os.path.join(WORKTREE_BASE, os.path.basename(repo), child_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Before adding one, not after: the copies this session is done with are what makes
    # room for this one, and reclaiming them later would mean nothing ever did.
    if session:
        reclaim_worktrees(session, SUBAGENT_WORKTREES_KEPT)
    # A copy whose directory went without git being told still holds its registration,
    # and those accumulate in .git/worktrees/ where only this ever looks.
    git(["worktree", "prune"], repo)
    # And the ones no session remembers any more: a deleted conversation takes its cards
    # with it, and the per-session rule above can no longer see what it left behind.
    sweep_orphan_copies(repo)
    # From HEAD, not from the working tree: a worktree cannot carry uncommitted changes,
    # and saying so beats a child hunting for code it cannot find.
    ok, out = git(["worktree", "add", "-b", branch, path, "HEAD"], repo)
    if not ok:
        return None, f"could not create the working copy: {out.splitlines()[-1] if out else 'git failed'}"
    # The commit it was cut from, kept as long as the copy is: it is what later tells a
    # branch carrying an axis from one carrying nothing, whatever the repository's HEAD
    # has moved on to by then.
    based, base = git(["rev-parse", "HEAD"], path)
    descriptor = {"repo": repo, "branch": branch, "path": path,
                  "base": base if based else ""}
    # Written before the copy is handed over, so no sweep can ever find it unowned: the
    # card that says the same thing lands a moment later, in the child's own thread.
    write_marker(marker_path(repo, child_key),
                  {**descriptor, "session": session, "pid": os.getpid()})
    return descriptor, ""


def finish_worktree(wt: dict, task: str, succeeded: bool) -> dict:
    """Commit the axis on its branch and keep the copy (its build tree); drop an empty branch."""
    path, branch, repo = wt["path"], wt["branch"], wt["repo"]
    base = wt.get("base", "")
    report = {"branch": branch, "path": path, "repo": repo, "base": base,
              "files_changed": [], "diffstat": "", "kept": True}

    if not succeeded:
        # Staged, not committed: a run that stopped early is continued in its copy, and
        # a commit taken mid-edit would have to be undone before that could start.
        git(["add", "-A"], path)
        ok, changed = git(["diff", "--cached", "--name-only"], path)
        report["files_changed"] = [f for f in changed.splitlines() if f] if ok else []
        ok, stat = git(["diff", "--cached", "--stat"], path)
        report["diffstat"] = stat if ok else ""
        report["note"] = (
            f"The run did not finish. Its copy is at {path}, with its uncommitted state "
            f"left as it was — that state is the only place that work exists.")
        return report

    files, diffstat = commit_copy(
        path, f"sub-agent {os.path.basename(path)}: {task[:120]}")
    report["files_changed"], report["diffstat"] = files, diffstat
    if files:
        report["committed"] = True
        report["note"] = (
            f"Its work is committed on {branch}, and the copy is still at {path}. Git "
            f"refuses to check that branch out anywhere else while the copy holds it, "
            f"so bring the work over with merge or cherry-pick, not checkout.")
        return report

    # Nothing written: the branch is still at the commit it was cut from and carries
    # nothing. Detaching first is what lets it go — git will not delete a branch a
    # worktree has checked out, and the copy itself is worth keeping either way. If
    # either step fails the branch is still there, and the report has to keep saying so.
    git(["checkout", "--detach"], path)
    if drop_branch_if_empty(repo, branch, base):
        report["branch"] = ""
        report["note"] = f"It changed no file, so it has no branch. Its copy is at {path}."
    else:
        report["note"] = (f"It changed no file. Its copy is at {path}, and {branch} is "
                          f"still where it was cut from.")
    return report
