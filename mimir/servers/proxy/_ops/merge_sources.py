"""What each proxy brings to a merge, read-only: its kept patch (baseline to best accepted
tree, from the snapshot store), its gain alone, the files two proxies share and the
conflicts between their patches, the gain they promise together. ``merge`` measures the
merge; the code map draws the forecast from here.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

from .._lib import tree_snapshot
from .._lib.layout import opt_runs_dir
from .._lib.ratchet import load_best, relative
from .._lib.store import load_opt_config, opt_git_dir, opt_ledger_file, workspace_root
from .eval_observe import read_ledger

_GIT_TIMEOUT_S = 60


class Conflict(Exception):
    """Both sides changed the same lines of a file."""


def source(name: str) -> tuple[dict | None, str]:
    """What *name* kept: its patch (baseline → best accepted tree), the files it touches and
    its gain alone. (None, why) when it has nothing to bring."""
    if "@" in name:
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
    rows = {r.get("run_id"): r for r in read_ledger(name)}
    base_value = (rows.get(base_run) or {}).get("primary_value")
    goal = cfg.get("primary_goal", "min")
    rel = relative(best.get("primary_value"), base_value, goal)
    if rel is None:
        return None, "its best or its baseline has no value"
    files = tree_snapshot.changed(opt_git_dir(), workspace_root(), cfg["baseline_id"], best["tree_snapshot"])
    if not files:
        return None, "its best tree changes no file"
    return {
        "proxy": name, "branch": cfg.get("branch", ""),
        "metric": cfg.get("primary_metric", "time_s"), "goal": goal,
        "suite": cfg.get("benchmark_name", ""),
        "base_value": base_value, "value": best.get("primary_value"), "run_id": best.get("run_id", ""),
        "relative": rel, "baseline_id": cfg["baseline_id"], "tree": best["tree_snapshot"],
        "files": files,
    }, ""


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


def _on_disk(rel: str) -> str | None:
    """The file as it is now, None when it does not exist; raises ``Conflict`` when it is
    not text, which no line merge can take."""
    try:
        with open(os.path.join(workspace_root(), rel), encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError:
        raise Conflict from None
    except OSError:
        return None


def apply(src: dict) -> list[str]:
    """Merge *src*'s patch into the files on disk: the files in conflict, and then nothing
    is written."""
    files: dict[str, str | None] = {}
    conflicts = []
    for rel in src["files"]:
        try:
            files[rel] = merge3(_on_disk(rel), _read(src["baseline_id"], rel), _read(src["tree"], rel))
        except Conflict:
            conflicts.append(rel)
    if conflicts:
        return conflicts
    root = workspace_root()
    for rel, text in files.items():
        path = os.path.join(root, rel)
        if text is None:
            if os.path.exists(path):
                os.remove(path)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return []


def pairs(sources: list[dict]) -> list[dict]:
    """For each two sources: the files both touch, and those whose two patches conflict."""
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
            out.append({"a": a["proxy"], "b": b["proxy"], "shared": shared, "conflicts": conflicts})
    return out


def expected(sources: list[dict]) -> float:
    """What the sources promise together if their gains multiply: lower is better."""
    value = 1.0
    for s in sources:
        value *= s["relative"]
    return round(value, 6)


def _live() -> list[str]:
    """Every proxy with a session in this workspace that is not itself a merge."""
    try:
        names = sorted(os.listdir(opt_runs_dir()))
    except OSError:
        return []
    return [n for n in names if os.path.isfile(opt_ledger_file(n))
            and not (load_opt_config(n) or {}).get("merge")]


def gather(names: list[str]) -> tuple[list[dict], list[dict], str]:
    """The sources *names* name ("*" for every proxy), those set aside and why, and a
    refusal when they cannot be measured together."""
    if names == ["*"]:
        names = _live()
    sources, refused = [], []
    for n in dict.fromkeys(names):
        src, why = source(n)
        if src:
            sources.append(src)
        else:
            refused.append({"proxy": n, "why": why})
    if len(sources) < 2:
        return sources, refused, "A merge needs two proxies at least that kept an accepted run."
    first = sources[0]
    for s in sources[1:]:
        for key, label in (("metric", "metric"), ("goal", "goal"), ("suite", "benchmark suite")):
            if s[key] != first[key]:
                return sources, refused, (f"{first['proxy']} and {s['proxy']} have a different {label} "
                                          f"({first[key]} / {s[key]}): one measurement cannot judge both.")
    sources.sort(key=lambda s: s["relative"])
    return sources, refused, ""


def public(src: dict) -> dict:
    return {**{k: src[k] for k in ("proxy", "branch", "run_id", "value", "base_value", "relative", "files")},
            "gain": round(1 - src["relative"], 6)}


def forecast(names: list[str]) -> dict:
    """What the map draws for proxies not merged yet: each one's gain alone, the files
    two of them share, the conflicts between them and the gain they promise together."""
    sources, refused, refusal = gather(names)
    if not sources:
        return {}
    return {"sources": [public(s) for s in sources], "refused": refused, "pairs": pairs(sources),
            "expected": expected(sources), "refusal": refusal}
