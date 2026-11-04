"""The optimisation session as the code map draws it: for each settled run, what it changed
against the incumbent it was measured on (files, and the dependencies those files add or
drop), its verdict, its axis, and its metric beside the incumbent's.

Rebuilt from the tree snapshots the ledger names, so a run ruled out still shows what it
tried. Cached on the ledger's mtime and its runs' profiles'. Any proxy's session can be
drawn, or all at once, and one frozen before a clean or a re-baseline (``freeze``).
"""
from __future__ import annotations

import os
import time

from ....shared import code_map
from ....shared.atomic import write_json
from .._lib import profiling, requests, tree_snapshot
from .._lib.ratchet import load_best, relative
from .._lib.layout import (history_dir, merge_file, opt_runs_dir, opt_session_runs_dir, read_json,
                           resolve_proxy_name)
from .._lib.store import load_opt_config, opt_git_dir, opt_ledger_file, workspace_root
from . import merge_sources
from .eval_observe import read_ledger

_CACHE: dict[str, tuple[tuple, dict]] = {}
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


def _profile(name: str, run_id: str, value, baseline) -> dict | None:
    """A run's profile with its shares in fractions of the baseline's metric, so two runs'
    bars compare; None when the run was not profiled. The sections taken apart (GPU,
    memory, counters, a kernel's) ride along even when the sampled CPU side could not run."""
    prof = read_json(profiling.profile_path(os.path.join(opt_session_runs_dir(name), run_id))) if run_id else None
    if not prof:
        return None
    scale = value / baseline if isinstance(value, (int, float)) and isinstance(baseline, (int, float)) and baseline else 1.0
    sections = {
        **({"gpu": _gpu(prof["gpu"], scale)} if prof.get("gpu") else {}),
        # Shares of the heap's peak, not of time: never scaled.
        **({"memory": {k: prof["memory"][k] for k in ("tool", "peak_mb", "files", "unavailable", "note")
                       if k in prof["memory"]}} if prof.get("memory") else {}),
        # Rates and percentages of peak, not shares of time: never scaled.
        **{k: prof[k] for k in ("counters", "kernel_detail") if prof.get(k)},
    }
    if prof.get("unavailable"):
        return {"unavailable": prof["unavailable"], "tool": prof.get("tool", ""), **sections}
    return {"tool": prof.get("tool", ""), "samples": prof.get("samples", 0), "case": prof.get("case", ""),
            "files": {f: round(v * scale, 5) for f, v in (prof.get("files") or {}).items()},
            "outside": round(sum((prof.get("outside") or {}).values()) * scale, 5),
            **({"functions": {k: {"self": round((v.get("self") or 0) * scale, 5),
                                  "total": round((v.get("total") or 0) * scale, 5),
                                  **({"line": v["line"]} if v.get("line") else {})}
                              for k, v in prof["functions"].items()}} if prof.get("functions") else {}),
            **sections,
            **({"note": prof["note"]} if prof.get("note") else {})}


def _gpu(g: dict, scale: float) -> dict:
    """A profile's GPU side in fractions of the baseline's time, its costliest kernels."""
    if g.get("unavailable"):
        return {"unavailable": g["unavailable"], "tool": g.get("tool", "")}
    return {"tool": g.get("tool", ""),
            "busy": round((g.get("busy") or 0) * scale, 5), "idle": round((g.get("idle") or 0) * scale, 5),
            "memcpy": {k: round(v * scale, 5) for k, v in (g.get("memcpy") or {}).items()},
            "files": {f: round(v * scale, 5) for f, v in (g.get("files") or {}).items()},
            "kernels": {k: {**v, "share": round((v.get("share") or 0) * scale, 5)}
                        for k, v in list((g.get("kernels") or {}).items())[:8]},
            # Sizes, not shares of time: never scaled.
            **{k: g[k] for k in ("memcpy_mb", "peak_device_mb") if g.get(k) is not None},
            **({"note": g["note"]} if g.get("note") else {})}


def _delta(value, incumbent) -> float | None:
    """The run's metric against its incumbent's, as a fraction; None when either is missing."""
    if not isinstance(value, (int, float)) or not isinstance(incumbent, (int, float)) or not incumbent:
        return None
    return (value - incumbent) / abs(incumbent)


def proxies() -> list[str]:
    """The proxies with a session in this workspace (those whose ledger exists), then the
    sessions frozen before a clean or a re-baseline."""
    try:
        names = sorted(os.listdir(opt_runs_dir()))
    except OSError:
        names = []
    return [n for n in names if os.path.isfile(opt_ledger_file(n))] + history()


def history() -> list[str]:
    """The frozen sessions, by key: ``<proxy>@<stamp>``."""
    try:
        names = sorted(os.listdir(history_dir()))
    except OSError:
        return []
    return [n[:-5] for n in names if n.endswith(".json") and "@" in n]


def _frozen_file(key: str) -> str | None:
    """Where the frozen session *key* is kept; None for a name that is not a key."""
    if "@" not in key or os.path.basename(key) != key:
        return None
    return os.path.join(history_dir(), key + ".json")


def freeze(name: str, why: str) -> str:
    """Keep *name*'s session as the map draws it, before *why* (cleaned, rebaselined) takes
    the ledger and snapshots it is rebuilt from. Its key, or "" when it had nothing to keep.
    Raises when it cannot be written: the caller must not take the state then."""
    value = session(name)
    if not value.get("steps") and (value.get("baseline") or {}).get("value") is None:
        return ""
    key = f"{name}@{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(history_dir(), exist_ok=True)
    # Under its key, steps included: the tab picks a session by the proxy its steps name.
    state = read_json(merge_file(name))
    write_json(os.path.join(history_dir(), key + ".json"), {
        **value, "proxy": key, "steps": [{**st, "proxy": key} for st in value.get("steps", [])],
        **({"merge": state} if state else {}),
        "frozen": {"proxy": name, "why": why, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}})
    return key


def _latest(names: list[str]) -> str:
    """The session that moved last: its ledger, or when it was frozen."""
    def mtime(n: str) -> float:
        try:
            return os.path.getmtime(_frozen_file(n) or opt_ledger_file(n))
        except OSError:
            return 0.0
    return max(names, key=mtime)


def _meta(name: str, active: str) -> dict:
    """One line of the tab's picker: where the session ran, its metric, how many runs it
    settled and its gain so far (1 - best/baseline, whatever the goal)."""
    frozen = _frozen_file(name)
    if frozen:
        s = read_json(frozen, {}) or {}
        base, goal = (s.get("baseline") or {}).get("value"), s.get("goal", "min")
        kept = [st["value"] for st in s.get("steps", []) if st.get("verdict") == "accept"]
        rel = min((r for r in (relative(v, base, goal) for v in kept) if r is not None), default=None)
        return {"name": name, "branch": "", "metric": s.get("metric", ""), "goal": goal,
                "steps": len(s.get("steps", [])), "gain": None if rel is None else round(1 - rel, 6),
                "active": False, "frozen": (s.get("frozen") or {}).get("why", "frozen"),
                "merge": bool(s.get("merge")), "sources": [x["proxy"] for x in (s.get("merge") or {}).get("sources", [])]}
    cfg = load_opt_config(name) or {}
    rows = read_ledger(name)
    base = next((r.get("primary_value") for r in rows if r.get("run_id") == cfg.get("baseline_run_id")), None)
    best = (load_best(name) or {}).get("primary_value")
    rel = relative(best, base, cfg.get("primary_goal", "min"))
    state = read_json(merge_file(name)) or {}
    return {"name": name, "branch": cfg.get("branch", ""), "metric": cfg.get("primary_metric", ""),
            "goal": cfg.get("primary_goal", "min"), "steps": max(0, len(rows) - bool(cfg.get("baseline_run_id"))),
            "gain": None if rel is None else round(1 - rel, 6), "active": name == active, "frozen": "",
            "merge": bool(cfg.get("merge")), "sources": [x["proxy"] for x in state.get("sources", [])],
            # A merge takes the proxies' own sessions, never another merge.
            "mergeable": not cfg.get("merge") and merge_sources.source(name)[0] is not None}


def answer(proxy_name: str = "", combined: bool = False, selected: list[str] | None = None) -> dict:
    """What the map tab is sent: the steps of *proxy_name* (else the active session's, else
    the latest session's), of the proxies *selected* together, or of every proxy at once
    when *combined*; with the proxies it can switch between, a line on each. Two proxies
    or more selected also bring what merging them promises (``forecast``)."""
    known = proxies()
    asked = requests_view()
    if not known:
        return {"steps": [], "requests": asked} if asked else {}
    active = resolve_proxy_name("") or ""
    picked = [n for n in dict.fromkeys(selected or []) if n in known]
    # A name the tab sends is one it was given; any other reads as the active session, and
    # without one (sub-agents measured, this session did not) as the latest.
    name = proxy_name if proxy_name in known else ("" if active else _latest(known))
    together = picked if len(picked) > 1 else known if combined else []
    value = merged(together) if together else session(name)
    # Always a list of steps, even empty: the tab still offers the other proxies.
    out = {"steps": [], **value, "proxies": known, "active": active,
           "proxies_meta": [_meta(n, active) for n in known],
           "combined": bool(together and value), "requests": asked}
    if len(picked) > 1:
        out["selected"] = picked
        out["forecast"] = merge_sources.forecast(picked)
    elif not together:
        # A merge session draws its sources, the whole and the one-at-a-time chain.
        state = value.get("merge") or (read_json(merge_file(value["proxy"])) if value.get("proxy") else None)
        if state:
            out["merge"] = state
    return out


def _steps_of(name: str) -> list[dict]:
    """A proxy's steps, its frozen sessions' first: a request tried before a clean or a
    re-baseline keeps its runs."""
    keys = [k for k in history() if k.rsplit("@", 1)[0] == name] + [name]
    return [st for k in keys for st in session(k).get("steps", [])]


def requests_view() -> list[dict]:
    """The R&D requests with the runs that tried them: those of their start's proxy whose
    axis names them, each against the value the request started from. A request with runs
    and no conclusion is being tried, whatever its status says."""
    out = []
    for row in requests.load():
        start = row.get("start") or {}
        mine = []
        attached = set(row.get("runs") or [])
        for st in _steps_of(start["proxy"]) if start.get("proxy") else []:
            tag = requests.axis_of(st.get("axis", ""))
            # By its axis tag, or attached to the request after the fact.
            if not (tag and tag[0] == row.get("id")) and st.get("run_id") not in attached:
                continue
            mine.append({"run_id": st["run_id"], "variant": tag[1] if tag else None, "axis": st["axis"],
                         "verdict": st["verdict"], "value": st["value"],
                         "delta": _delta(st["value"], start.get("value"))})
        status = row.get("status", "open")
        if mine and not row.get("conclusion") and status in ("open", "studying", "proposed"):
            status = "trying"
        out.append({**{k: row.get(k) for k in ("id", "token", "ts", "mode", "kind", "targets", "idea", "variants",
                                                "source", "constraints", "summary", "conclusion", "session")},
                    "status": status, "proxy": start.get("proxy", ""), "metric": start.get("metric", ""),
                    "start_value": start.get("value"), "steps": mine})
    return out


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


def session(proxy_name: str = "") -> dict:
    """The session's steps of *proxy_name* (else the active one's), or {} when there is none.
    A frozen session's key reads it as it was frozen."""
    frozen = _frozen_file(proxy_name)
    if frozen:
        return read_json(frozen, {}) or {}
    name = resolve_proxy_name(proxy_name)
    if not name:
        return {}
    try:
        mtime = (os.path.getmtime(opt_ledger_file(name)), _profiles_stamp(name))
    except OSError:
        return {}
    cached = _CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    cfg = load_opt_config(name) or {}
    rows = read_ledger(name)
    root = workspace_root()
    by_run = {r.get("run_id"): r for r in rows}
    memo: dict[str, dict] = {}
    baseline_tree = cfg.get("baseline_id", "")
    baseline_run = cfg.get("baseline_run_id", "")
    steps: list[dict] = []
    incumbent = by_run.get(baseline_run)
    base_value = (incumbent or {}).get("primary_value")
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
            "proxy": name,
            "ts": row.get("ts", ""),
            "run_id": row.get("run_id", ""),
            "axis": row.get("axis", ""),
            "verdict": row.get("verdict", ""),
            "value": row.get("primary_value"),
            "incumbent_run_id": (inc or {}).get("run_id", ""),
            "incumbent_value": (inc or {}).get("primary_value"),
            "delta": _delta(row.get("primary_value"), (inc or {}).get("primary_value")),
            "peak_rss_mb": row.get("peak_rss_mb"),
            "incumbent_peak_rss_mb": (inc or {}).get("peak_rss_mb"),
            "peak_rss_total_mb": row.get("peak_rss_total_mb"),
            "incumbent_peak_rss_total_mb": (inc or {}).get("peak_rss_total_mb"),
            "changed": changed or [],
            "added": [{k: now[e][k] for k in _EDGE_FIELDS} for e in sorted(set(now) - set(then))],
            "removed": [{k: then[e][k] for k in _EDGE_FIELDS} for e in sorted(set(then) - set(now))],
            "profile": _profile(name, row.get("run_id", ""), row.get("primary_value"), base_value),
        })
    value = {
        "proxy": name,
        "metric": cfg.get("primary_metric", "time_s"),
        "goal": cfg.get("primary_goal", "min"),
        "baseline": {"run_id": baseline_run, "value": base_value,
                     "peak_rss_mb": (incumbent or {}).get("peak_rss_mb"),
                     "profile": _profile(name, baseline_run, base_value, base_value)},
        "steps": steps,
    }
    _CACHE[name] = (mtime, value)
    return value


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
            steps.append({**st, "value": now, "incumbent_value": then, "delta": _delta(now, then),
                          "metric": s["metric"]})
    steps.sort(key=lambda st: st.get("ts") or "")
    for i, st in enumerate(steps):
        st["n"] = i + 1
    if not steps:
        return {}
    return {
        "proxy": ", ".join(s["proxy"] for s in sessions),
        "metric": "vs baseline",
        "goal": "min",
        "baseline": {"run_id": "", "value": 1.0, "profile": None},
        "steps": steps,
    }
