"""The optimisation session as the code map draws it: for each settled run, what it changed
against the incumbent it was measured on (files, and the dependencies those files add or
drop), its verdict, its axis, and its metric beside the incumbent's.

Rebuilt from the tree snapshots the ledger names, so a run ruled out still shows what it
tried. Cached on the ledger's mtime: it only moves when a run settles.
"""
from __future__ import annotations

import os

from ....shared import code_map
from .._lib import tree_snapshot
from .._lib.layout import resolve_proxy_name
from .._lib.store import load_opt_config, opt_git_dir, opt_ledger_file, workspace_root
from .eval_observe import read_ledger

_CACHE: dict = {"key": None, "value": {}}
_EDGE_FIELDS = ("from", "to", "line", "kind", "via", "sure", "breaks")


def _edges_at(root: str, snapshot_id: str, memo: dict[str, dict]) -> dict:
    """The code map's edges with the tracked files as *snapshot_id* holds them, keyed by
    (from, to, kind); the rest of the tree is read from disk."""
    if snapshot_id not in memo:
        git = opt_git_dir()
        held = tree_snapshot.listing(git, root, snapshot_id) or []
        texts = {rel: tree_snapshot.read(git, root, snapshot_id, rel) for rel in held}
        m = code_map.scan(root, overrides={rel: t for rel, t in texts.items()
                                           if code_map.language_of(rel) and t is not None})
        memo[snapshot_id] = {(e["from"], e["to"], e["kind"]): e for e in m["edges"]}
    return memo[snapshot_id]


def _delta(value, incumbent) -> float | None:
    """The run's metric against its incumbent's, as a fraction; None when either is missing."""
    if not isinstance(value, (int, float)) or not isinstance(incumbent, (int, float)) or not incumbent:
        return None
    return (value - incumbent) / abs(incumbent)


def session(proxy_name: str = "") -> dict:
    """The current session's steps, or {} when there is none."""
    name = resolve_proxy_name(proxy_name)
    if not name:
        return {}
    try:
        key = (name, os.path.getmtime(opt_ledger_file(name)))
    except OSError:
        return {}
    if key == _CACHE["key"]:
        return _CACHE["value"]
    cfg = load_opt_config(name) or {}
    rows = read_ledger(name)
    root = workspace_root()
    by_run = {r.get("run_id"): r for r in rows}
    memo: dict[str, dict] = {}
    baseline_tree = cfg.get("baseline_id", "")
    baseline_run = cfg.get("baseline_run_id", "")
    steps: list[dict] = []
    incumbent = by_run.get(baseline_run)
    for i, row in enumerate(rows):
        if row.get("run_id") == baseline_run:
            continue
        before = by_run.get(rows[i - 1].get("best_run_id")) if i else None
        inc = before or incumbent
        inc_tree = (inc or {}).get("launch_tree") or baseline_tree
        tree = row.get("launch_tree") or ""
        changed = tree_snapshot.changed(opt_git_dir(), root, inc_tree, tree) if tree else None
        now, then = (_edges_at(root, tree, memo), _edges_at(root, inc_tree, memo)) if tree else ({}, {})
        steps.append({
            "n": len(steps) + 1,
            "run_id": row.get("run_id", ""),
            "axis": row.get("axis", ""),
            "verdict": row.get("verdict", ""),
            "value": row.get("primary_value"),
            "incumbent_run_id": (inc or {}).get("run_id", ""),
            "incumbent_value": (inc or {}).get("primary_value"),
            "delta": _delta(row.get("primary_value"), (inc or {}).get("primary_value")),
            "changed": changed or [],
            "added": [{k: now[e][k] for k in _EDGE_FIELDS} for e in sorted(set(now) - set(then))],
            "removed": [{k: then[e][k] for k in _EDGE_FIELDS} for e in sorted(set(then) - set(now))],
        })
    value = {
        "proxy": name,
        "metric": cfg.get("primary_metric", "time_s"),
        "goal": cfg.get("primary_goal", "min"),
        "baseline": {"run_id": baseline_run,
                     "value": (by_run.get(baseline_run) or {}).get("primary_value")},
        "steps": steps,
    }
    _CACHE.update({"key": key, "value": value})
    return value
