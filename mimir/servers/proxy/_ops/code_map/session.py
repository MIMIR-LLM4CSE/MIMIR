"""One optimisation session as the map draws it: for each settled run, what it changed
against the incumbent it was measured on (files, and the dependencies those files add or
drop), its verdict, its axis, and its metric beside the incumbent's.

Rebuilt from the tree snapshots the ledger names, so a run ruled out still shows what it
tried; cached on the ledger's mtime and its runs' profiles'. A frozen session's key reads
it as it was kept (``freeze``).
"""
from __future__ import annotations

import os
import threading

from .....shared import code_map
from ..._lib import history, profiling, tree_snapshot
from ..._lib.layout import merge_file, opt_session_runs_dir, resolve_proxy_name
from .....shared.atomic import read_json
from ..._lib.ratchet import baseline_value, read_ledger, relative
from ..._lib.store import load_opt_config, opt_git_dir, opt_ledger_file, workspace_root
from . import profile

_CACHE: dict[str, tuple[tuple, dict]] = {}
# One build per proxy at a time: a call that arrives during the warm-up at start waits for
# it rather than reading every snapshot again.
_BUILDING: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()
_EDGE_FIELDS = ("from", "to", "line", "kind", "via", "sure", "breaks")


def delta(value, incumbent) -> float | None:
    """The run's metric against its incumbent's, as a fraction; None when either is missing."""
    if not isinstance(value, (int, float)) or not isinstance(incumbent, (int, float)) or not incumbent:
        return None
    return (value - incumbent) / abs(incumbent)


def session(proxy_name: str = "") -> dict:
    """The steps of *proxy_name*'s session (else the active one's), or {} when there is none."""
    if history.is_key(proxy_name):
        return history.read(proxy_name)
    name = resolve_proxy_name(proxy_name)
    if not name:
        return {}
    try:
        stamp = (os.path.getmtime(opt_ledger_file(name)), _profiles_stamp(name))
    except OSError:
        return {}
    with _GUARD:
        lock = _BUILDING.setdefault(name, threading.Lock())
    with lock:
        cached = _CACHE.get(name)
        if cached and cached[0] == stamp:
            return cached[1]
        value = _build(name)
        _CACHE[name] = (stamp, value)
        return value



def _build(name: str) -> dict:
    cfg = load_opt_config(name) or {}
    rows = read_ledger(name)
    root = workspace_root()
    by_run = {r.get("run_id"): r for r in rows}
    memo: dict[str, dict] = {}
    baseline_run = cfg.get("baseline_run_id", "")
    first = by_run.get(baseline_run)
    base_value = baseline_value(cfg, rows)
    steps: list[dict] = []
    for i, row in enumerate(rows):
        if row.get("run_id") == baseline_run:
            continue
        inc = (by_run.get(rows[i - 1].get("best_run_id")) if i else None) or first or {}
        inc_tree = inc.get("launch_tree") or cfg.get("baseline_id", "")
        tree = row.get("launch_tree") or ""
        changed = tree_snapshot.changed(opt_git_dir(), root, inc_tree, tree) if tree else None
        now, then = (_edges_at(root, tree, memo), _edges_at(root, inc_tree, memo)) if tree else ({}, {})
        steps.append({
            "n": len(steps) + 1,
            "proxy": name,
            "ts": row.get("ts", ""),
            "run_id": row.get("run_id", ""),
            "axis": row.get("axis", ""),
            "verdict": row.get("verdict", ""),
            "value": row.get("primary_value"),
            "incumbent_run_id": inc.get("run_id", ""),
            "incumbent_value": inc.get("primary_value"),
            "delta": delta(row.get("primary_value"), inc.get("primary_value")),
            "peak_rss_mb": row.get("peak_rss_mb"),
            "incumbent_peak_rss_mb": inc.get("peak_rss_mb"),
            "peak_rss_total_mb": row.get("peak_rss_total_mb"),
            "incumbent_peak_rss_total_mb": inc.get("peak_rss_total_mb"),
            "changed": changed or [],
            "added": [{k: now[e][k] for k in _EDGE_FIELDS} for e in sorted(set(now) - set(then))],
            "removed": [{k: then[e][k] for k in _EDGE_FIELDS} for e in sorted(set(then) - set(now))],
            "profile": profile.scaled(name, row.get("run_id", ""), row.get("primary_value"), base_value),
        })
    return {
        "proxy": name,
        "metric": cfg.get("primary_metric", "time_s"),
        "goal": cfg.get("primary_goal", "min"),
        "baseline": {"run_id": baseline_run, "value": base_value,
                     "peak_rss_mb": (first or {}).get("peak_rss_mb"),
                     "profile": profile.scaled(name, baseline_run, base_value, base_value)},
        "steps": steps,
    }


def _edges_at(root: str, snapshot_id: str, memo: dict[str, dict]) -> dict:
    """The code map's edges with the tracked files as *snapshot_id* holds them, keyed by
    (from, to, kind); the rest of the tree is read from disk."""
    if snapshot_id not in memo:
        texts = tree_snapshot.read_all(opt_git_dir(), root, snapshot_id)
        m = code_map.scan(root, overrides={rel: t for rel, t in texts.items() if code_map.language_of(rel)})
        memo[snapshot_id] = {(e["from"], e["to"], e["kind"]): e for e in m["edges"]}
    return memo[snapshot_id]


def _profiles_stamp(name: str) -> tuple[int, float]:
    """How many of the session's runs hold a profile and when the latest was written: a
    profile taken after its run's verdict leaves the ledger as it was."""
    runs = opt_session_runs_dir(name)
    try:
        dirs = os.listdir(runs)
    except OSError:
        return 0, 0.0
    stamps = []
    for d in dirs:
        try:
            stamps.append(os.path.getmtime(profiling.profile_path(os.path.join(runs, d))))
        except OSError:
            pass
    return len(stamps), max(stamps, default=0.0)


def merged(names: list[str]) -> dict:
    """Every proxy's steps as one session, in the order they settled. Values are each proxy's
    metric against its own baseline (1.0), so the strip draws one line across metrics; what a
    step changed is still against its own proxy's incumbent, and "every kept step" marks all
    the proxies' kept changes together. No baseline profile: each proxy has its own."""
    sessions = [s for s in (session(n) for n in names) if s.get("steps")]
    steps: list[dict] = []
    for s in sessions:
        base, goal = s["baseline"]["value"], s["goal"]
        for st in s["steps"]:
            now = relative(st["value"], base, goal)
            then = relative(st["incumbent_value"], base, goal)
            steps.append({**st, "value": now, "incumbent_value": then, "delta": delta(now, then),
                          "metric": s["metric"]})
    if not steps:
        return {}
    steps.sort(key=lambda st: st.get("ts") or "")
    for i, st in enumerate(steps):
        st["n"] = i + 1
    return {
        "proxy": ", ".join(s["proxy"] for s in sessions),
        "metric": "vs baseline",
        "goal": "min",
        "baseline": {"run_id": "", "value": 1.0, "profile": None},
        "steps": steps,
    }


def freeze(name: str, why: str) -> str:
    """Keep *name*'s session as the map draws it, merge state included, before *why*
    (cleaned, rebaselined) takes the ledger and snapshots it is rebuilt from. Its key, or ""
    when it had nothing to keep. Raises when it cannot be written: the caller must not take
    the state then."""
    value = session(name)
    if not value.get("steps") and (value.get("baseline") or {}).get("value") is None:
        return ""
    return history.write(name, why, value, read_json(merge_file(name)))
