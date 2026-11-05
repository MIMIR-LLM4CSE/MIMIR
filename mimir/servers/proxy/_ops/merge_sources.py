"""What each proxy brings to a merge: its kept patch (baseline to best accepted tree, from
the snapshot store), its gain alone, the files two proxies share and the conflicts
between their patches; and, in the merge's copy, that patch as a commit git can pick
(``kept_commit``). ``merge`` runs the merge; the code map draws the forecast from here.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime

from .._lib import history, tree_snapshot
from .._lib.ratchet import baseline_value, load_best, read_ledger, relative
from .._lib.store import load_opt_config, opt_git_dir, opt_sessions, workspace_root

_GIT_TIMEOUT_S = 60


class Conflict(Exception):
    """Both sides changed the same lines of a file."""


def source(name: str) -> tuple[dict | None, str]:
    """What *name* kept: its patch (baseline → best accepted tree), the files it touches and
    its gain alone. (None, why) when it has nothing to bring."""
    if history.is_key(name):
        return None, "a frozen session: its trees are no longer kept"
    cfg = load_opt_config(name) or {}
    if not cfg.get("baseline_id"):
        return None, "no optimisation session"
    best = load_best(name) or {}
    base_run = cfg.get("baseline_run_id", "")
    if not base_run:
        return None, "its baseline was never measured"
    if not best.get("tree_snapshot") or best.get("run_id") == base_run:
        return None, "no run accepted beyond its baseline"
    base_value = baseline_value(cfg, read_ledger(name))
    goal = cfg.get("primary_goal", "min")
    rel = relative(best.get("primary_value"), base_value, goal)
    if rel is None:
        return None, "its best or its baseline has no value"
    files = own(cfg, tree_snapshot.changed(opt_git_dir(), workspace_root(), cfg["baseline_id"], best["tree_snapshot"]) or [])
    if not files:
        return None, "its best tree changes no file"
    return {
        "proxy": name, "branch": cfg.get("branch", ""),
        "metric": cfg.get("primary_metric", "time_s"), "goal": goal,
        "suite": cfg.get("benchmark_name", ""), "requirements": cfg.get("requirements") or [],
        "base_value": base_value, "value": best.get("primary_value"), "run_id": best.get("run_id", ""),
        "relative": rel, "baseline_id": cfg["baseline_id"], "tree": best["tree_snapshot"],
        "files": files, "root": root_of(cfg), "span": span(read_ledger(name)),
    }, ""


def span(rows: list[dict]) -> list[str]:
    """When a session ran: its first and last ledger rows' times, [] without any."""
    ts = sorted(r["ts"] for r in rows if isinstance(r.get("ts"), str) and r["ts"])
    return [ts[0], ts[-1]] if ts else []


def together(a: dict, b: dict) -> bool:
    """Whether *a* and *b* ran at the same time in the same tree: a snapshot of one then
    held, in a file both change, what the other had on disk, so its patch and its gain
    alone may hold the other's work."""
    if not (a.get("root") and a.get("root") == b.get("root") and a.get("span") and b.get("span")):
        return False
    try:
        (a0, a1), (b0, b1) = ([datetime.fromisoformat(t) for t in x["span"]] for x in (a, b))
    except ValueError:
        return False
    return a0 <= b1 and b0 <= a1


def root_of(cfg: dict) -> str:
    """The tree a session's snapshot paths are relative to: recorded at init, else read
    off a tracked path whose end is one of the baseline's files."""
    if cfg.get("root"):
        return cfg["root"]
    held = tree_snapshot.listing(opt_git_dir(), workspace_root(), cfg.get("baseline_id", "")) or []
    for path in cfg.get("optimize_paths") or []:
        for rel in held:
            if path.endswith(os.sep + rel):
                return path[: -len(rel) - 1]
    return ""


def own(cfg: dict, files: list[str]) -> list[str]:
    """Of *files* (snapshot paths), those under the session's own optimize_paths: a
    snapshot taken while copies shared the store's index also holds the paths another
    proxy had staged, which are not its work."""
    root = root_of(cfg)
    if not root:
        return files
    mine = [os.path.relpath(p, root).replace(os.sep, "/") for p in cfg.get("optimize_paths") or []]
    return [f for f in files if any(f == m or f.startswith(m + "/") for m in mine)]


def _read(snapshot_id: str, rel: str) -> str | None:
    return tree_snapshot.read(opt_git_dir(), workspace_root(), snapshot_id, rel)


def merge3(ours: str | None, base: str | None, theirs: str | None) -> str | None:
    """One file merged three ways: the result (None for a deleted file), or raises
    ``Conflict`` when the two sides cannot both be kept."""
    if ours == theirs or theirs == base:
        return ours
    if ours == base:
        return theirs
    if ours is None or base is None or theirs is None:
        raise Conflict  # one side added or deleted it, the other changed it
    if not tree_snapshot.git_available():
        raise Conflict  # no line merge without git: both changes are kept apart
    with tempfile.TemporaryDirectory() as tmp:
        names = []
        for label, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            path = os.path.join(tmp, label)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            names.append(path)
        try:
            r = subprocess.run(["git", "merge-file", "-p", *names], capture_output=True,
                               text=True, timeout=_GIT_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            raise Conflict from None
    if r.returncode != 0:
        raise Conflict
    return r.stdout


def pairs(sources: list[dict]) -> list[dict]:
    """For each two sources: the files both touch, those whose two patches conflict, and
    whether they ran at the same time in the same tree (``together``)."""
    out = []
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            shared = sorted(set(a["files"]) & set(b["files"]))
            conflicts = []
            for rel in shared:
                try:
                    merge3(_read(a["tree"], rel), _read(b["baseline_id"], rel), _read(b["tree"], rel))
                except Conflict:
                    conflicts.append(rel)
            out.append({"a": a["proxy"], "b": b["proxy"], "shared": shared, "conflicts": conflicts,
                        "together": bool(shared) and together(a, b)})
    return out


def gather(names: list[str]) -> tuple[list[dict], list[dict], str]:
    """The sources *names* name ("*" for every proxy), those set aside and why, and a
    refusal when fewer than two bring anything. Each is measured on its own suite in the
    merge, so neither their metric nor their suite need agree."""
    if names == ["*"]:
        # Every proxy with a session that is not itself a merge.
        names = [n for n in opt_sessions() if not (load_opt_config(n) or {}).get("merge")]
    sources, refused = [], []
    for n in dict.fromkeys(names):
        src, why = source(n)
        if src:
            sources.append(src)
        else:
            refused.append({"proxy": n, "why": why})
    if len(sources) < 2:
        return sources, refused, "A merge needs two proxies at least that kept an accepted run."
    sources.sort(key=lambda s: s["relative"])
    return sources, refused, ""


def public(src: dict) -> dict:
    return {**{k: src[k] for k in ("proxy", "branch", "suite", "metric", "run_id", "value", "base_value",
                                   "relative", "files")},
            "gain": round(1 - src["relative"], 6)}


def forecast(names: list[str]) -> dict:
    """What the map draws for proxies not merged yet: each one's gain alone, the files
    two of them share and the conflicts to resolve between them."""
    sources, refused, refusal = gather(names)
    if not sources:
        return {}
    return {"sources": [public(s) for s in sources], "refused": refused, "pairs": pairs(sources),
            "refusal": refusal}


# ── the patch as a commit, in the merge's copy ────────────────────────────────

def git(*args: str, index: str = "", stdin: str | None = None) -> subprocess.CompletedProcess:
    """git in the workspace's own repository (the merge's copy), *index* as its index
    when given, never a shell."""
    env = {**os.environ, **({"GIT_INDEX_FILE": index} if index else {})}
    return subprocess.run(["git", "-c", "user.name=MIMIR merge", "-c", "user.email=merge@mimir.invalid",
                           "-c", "commit.gpgsign=false", *args],
                          cwd=workspace_root(), env=env, input=stdin, capture_output=True,
                          text=True, timeout=_GIT_TIMEOUT_S)


def _put(index: str, path: str, text: str | None) -> bool:
    """*path* in *index* holding *text*, or gone from it for None."""
    if text is None:
        return git("update-index", "--force-remove", "--", path, index=index).returncode == 0
    blob = git("hash-object", "-w", "--stdin", stdin=text)
    if blob.returncode != 0:
        return False
    staged = git("ls-files", "-s", "--", path, index=index).stdout.split()
    mode = staged[0] if staged else "100644"
    return git("update-index", "--add", "--cacheinfo", f"{mode},{blob.stdout.strip()},{path}",
               index=index).returncode == 0


def already_in(src: dict) -> bool:
    """Whether this copy's HEAD already holds *src*'s kept files, as its branch does when
    that work was committed on it."""
    at = git("rev-parse", "--show-prefix").stdout.strip()
    for rel in src["files"]:
        here = git("show", f"HEAD:{at}{rel}")
        if (here.stdout if here.returncode == 0 else None) != _read(src["tree"], rel):
            return False
    return bool(src["files"])


def kept_commit(src: dict) -> tuple[str, str]:
    """*src*'s kept patch as a commit over this copy's HEAD, with no branch: (sha, "") or
    ("", why). Its parent holds *src*'s files as its baseline had them and the commit as
    its best tree has them, so picking it is a three-way merge whose base is the proxy's
    own baseline, whatever the files hold here now."""
    head = git("rev-parse", "HEAD")
    prefix = git("rev-parse", "--show-prefix")
    if head.returncode != 0 or prefix.returncode != 0:
        return "", "this copy is not a git repository"
    at = prefix.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="mimir-merge-") as tmp:
        index = os.path.join(tmp, "index")
        if git("read-tree", "HEAD", index=index).returncode != 0:
            return "", "HEAD cannot be read"
        parent = head.stdout.strip()
        for snapshot, message in ((src["baseline_id"], f"{src['proxy']}: its baseline"),
                                  (src["tree"], f"kept state of {src['proxy']} (run {src['run_id']}, "
                                                f"{round((1 - src['relative']) * 100, 1)}% better alone)")):
            if not all(_put(index, at + rel, _read(snapshot, rel)) for rel in src["files"]):
                return "", f"{src['proxy']}'s files cannot be written to the index"
            tree = git("write-tree", index=index)
            commit = git("commit-tree", tree.stdout.strip(), "-p", parent, "-m", message)
            if tree.returncode != 0 or commit.returncode != 0:
                return "", f"{src['proxy']}'s commit cannot be written"
            parent = commit.stdout.strip()
    return parent, ""
