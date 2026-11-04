"""proxy_merge ops: plan, start, next, report. The kept optimisations of several proxies,
each a patch from its baseline to its best accepted tree read from the snapshot store,
merged three ways file by file in one copy and measured there: all at once first, then
one by one when the whole falls short of what the parts promised. A source in conflict
writes nothing and is set aside. The merge is an ordinary proxy session, ``merge-<stamp>``,
whose ``merge.json`` holds the sources, the phase and what each did to the whole.
"""
from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from . import check_name, err, ok, with_next
from ....shared.atomic import write_json
from ....shared.state_paths import active_session_id
from .._lib import tree_snapshot
from .._lib.layout import merge_file, opt_runs_dir, read_json, resolve_proxy_name
from .._lib.procs import opt_active_run_dir, run_state
from .._lib.ratchet import effective_min_improvement, is_improvement, load_best, relative
from .._lib.store import (load_opt_config, load_registry, opt_git_dir, opt_ledger_file, registry_lock,
                          save_opt_config, save_registry, workspace_root)
from . import eval_observe, eval_session

_GIT_TIMEOUT_S = 60
_TOOLS = ["proxy_merge", "proxy_eval", "proxy_eval_status"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── what one source brings ────────────────────────────────────────────────────

def _root_of(cfg: dict) -> str:
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


def source(name: str) -> tuple[dict | None, str]:
    """What *name* kept: its patch (baseline → best accepted tree), the files it touches and
    its gain alone. (None, why) when it has nothing to bring."""
    cfg = load_opt_config(name) or {}
    if not cfg.get("baseline_id"):
        return None, "no optimisation session"
    best = load_best(name) or {}
    base_run = cfg.get("baseline_run_id", "")
    if not base_run:
        return None, "its baseline was never measured"
    if not best.get("tree_snapshot") or best.get("run_id") == base_run:
        return None, "no run accepted beyond its baseline"
    rows = {r.get("run_id"): r for r in eval_observe.read_ledger(name)}
    base_value = (rows.get(base_run) or {}).get("primary_value")
    goal = cfg.get("primary_goal", "min")
    rel = relative(best.get("primary_value"), base_value, goal)
    if rel is None:
        return None, "its best or its baseline has no value"
    files = tree_snapshot.changed(opt_git_dir(), workspace_root(), cfg["baseline_id"], best["tree_snapshot"])
    if not files:
        return None, "its best tree changes no file"
    return {
        "proxy": name, "branch": cfg.get("branch", ""), "root": _root_of(cfg),
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
        raise Conflict
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


class Conflict(Exception):
    """Both sides changed the same lines of a file."""


def _patched(src: dict, current) -> tuple[dict[str, str | None], list[str]]:
    """*src*'s patch over the files as *current(rel)* gives them: the files to write
    (None deletes) and those in conflict."""
    out: dict[str, str | None] = {}
    conflicts = []
    for rel in src["files"]:
        try:
            out[rel] = merge3(current(rel), _read(src["baseline_id"], rel), _read(src["tree"], rel))
        except Conflict:
            conflicts.append(rel)
    return out, conflicts


def _on_disk(rel: str) -> str | None:
    try:
        with open(os.path.join(workspace_root(), rel), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def apply(src: dict) -> list[str]:
    """Merge *src*'s patch into the files on disk: the files in conflict, and then nothing
    is written."""
    files, conflicts = _patched(src, _on_disk)
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


def _pairs(sources: list[dict]) -> list[dict]:
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


def _expected(sources: list[dict]) -> float:
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
    if not names or names == ["*"]:
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


def _public(src: dict) -> dict:
    return {k: src[k] for k in ("proxy", "branch", "run_id", "value", "base_value", "relative", "files")} | {
        "gain": round(1 - src["relative"], 6)}


def forecast(names: list[str]) -> dict:
    """What the map draws for proxies not merged yet: each one's gain alone, the files
    two of them share, the conflicts between them and the gain they promise together."""
    sources, refused, refusal = gather(names)
    if not sources:
        return {}
    expected = _expected(sources)
    return {"sources": [_public(s) for s in sources], "refused": refused, "pairs": _pairs(sources),
            "expected": expected, "refusal": refusal}


# ── ops ───────────────────────────────────────────────────────────────────────

def _brief(names: list[str]) -> str:
    listed = ", ".join(names)
    return (
        f"Merge the kept optimisations of the proxies {listed} and measure the result. In your "
        f"copy: proxy_merge(op='start', proxies={names!r}, confirm=True); then follow its "
        "next_step — proxy_merge(op='next', confirm=True) and the proxy_eval(op='run', axis=..., "
        "confirm=True) it names, in turn — until it says the merge is done; edit no file "
        "yourself. Then answer with proxy_merge(op='report'): the gain expected and measured, "
        "the proxies kept and those set aside, and why."
    )


def plan(names: list[str]) -> dict:
    sources, refused, refusal = gather(names)
    if refusal:
        return err(refusal, refused=refused)
    picked = [s["proxy"] for s in sources]
    pairs = _pairs(sources)
    expected = _expected(sources)
    return ok(with_next({
        "sources": [_public(s) for s in sources],
        "refused": refused,
        "pairs": pairs,
        "expected": {"relative": expected, "gain": round(1 - expected, 6)},
        "order": picked,
        "brief": _brief(picked),
        "note": ("The expected gain assumes the gains multiply. All the sources are applied and "
                 "measured at once first; when that falls short of the expected gain beyond the "
                 "noise, they are measured again one at a time, best alone first, and one that "
                 "slows the whole down is set aside. Pairs with conflicts cannot both be kept."),
    }, f"spawn_agent(task=<brief>, tools={_TOOLS!r}, background=True): the merge runs in a "
       "copy of its own, on a branch you merge once it is measured"))


def _in_copy() -> bool:
    """Whether this server works for a sub-agent, in a copy of its own."""
    return "/subagents/" in "/" + active_session_id()


def _moved(value, src_root: str, here: str):
    """*value* with every path under *src_root* moved under *here*."""
    if isinstance(value, str):
        if value == src_root or value.startswith(src_root.rstrip(os.sep) + os.sep):
            return here + value[len(src_root):]
        return value.replace(src_root.rstrip(os.sep) + os.sep, here.rstrip(os.sep) + os.sep)
    if isinstance(value, list):
        return [_moved(v, src_root, here) for v in value]
    if isinstance(value, dict):
        return {k: _moved(v, src_root, here) for k, v in value.items()}
    return value


def start(names: list[str], name: str = "") -> dict:
    if not _in_copy():
        return err("A merge is measured in a copy of its own, and this is the main workspace.",
                   hint="proxy_merge(op='plan', proxies=[...]) gives the brief to hand to spawn_agent.")
    sources, refused, refusal = gather(names)
    if refusal:
        return err(refusal, refused=refused)
    name = name or f"merge-{time.strftime('%Y%m%d-%H%M%S')}"
    bad = check_name("name", name)
    if bad:
        return bad
    here = workspace_root()
    first = sources[0]
    cfg = load_opt_config(first["proxy"]) or {}
    entry = (load_registry() or {}).get(first["proxy"])
    if not entry:
        return err(f"{first['proxy']} is no longer registered: its harness cannot be found.")
    if not first["root"]:
        return err(f"Where {first['proxy']}'s files lived cannot be told, so its paths cannot be moved here.")
    harness = _moved(cfg.get("proxy_source_path", ""), first["root"], here)
    if not os.path.isfile(harness):
        return err(f"The harness {os.path.relpath(harness, here)} is not in this copy: it was "
                   f"committed only on {first['branch'] or 'its own copy'}.",
                   hint=f"git checkout {first['branch'] or '<branch>'} -- {os.path.relpath(harness, here)}, "
                        "then proxy_merge(op='start') again.")
    paths = []
    for s in sources:
        own = load_opt_config(s["proxy"]) or {}
        paths += [_moved(p, s["root"], here) for p in own.get("optimize_paths") or [] if s["root"]]
    paths = sorted(set(paths))
    with registry_lock(create=True):
        reg = load_registry() or {}
        if name in reg:
            return err(f"A proxy named {name} is already registered.")
        reg[name] = {**_moved(copy.deepcopy(entry), first["root"], here), "name": name,
                     "registered_at": _now(), "merged_from": [s["proxy"] for s in sources]}
        reg[name].pop("updated_at", None)
        save_registry(reg)
    res = eval_session.init(
        name, first["suite"], cfg.get("requirements") or [], harness, paths,
        _moved(cfg.get("python_executable", ""), first["root"], here), cfg.get("max_hours", 0.0),
        first["metric"], first["goal"], cfg.get("min_improvement", 0.02),
        max_stall=len(sources) + 3, convergence=cfg.get("convergence"), repeat=cfg.get("repeat", 0))
    if res.get("status") != "ok":
        return res
    mine = load_opt_config(name)
    mine["merge"] = True
    save_opt_config(mine)
    _save(name, {
        "name": name, "branch": mine.get("branch", ""), "started_at": _now(),
        "sources": sources, "refused": refused, "pairs": _pairs(sources),
        "expected": _expected(sources), "phase": "baseline", "pending": None,
        "base_value": None, "all": None, "steps": [], "final": None,
    })
    return ok(with_next({"proxy_name": name, "sources": [_public(s) for s in sources],
                         "refused": refused, "expected": _expected(sources)},
                        "proxy_merge(op='next', confirm=True)"))


def _load(name: str) -> dict:
    return read_json(merge_file(name), {}) or {}


def _save(name: str, state: dict) -> None:
    write_json(merge_file(name), state, indent=2)


def _settled(name: str, pending: dict) -> dict | str | None:
    """The run settled since *pending* was set, "running" while one is under way, None
    when none was launched."""
    rows = eval_observe.read_ledger(name)
    if len(rows) <= pending["ledger"]:
        active = opt_active_run_dir(name)
        if active and run_state(active)["state"] in ("running", "pending"):
            return "running"
        if active:
            # Ended, not settled yet: settling is the results read's, whoever comes first.
            eval_observe.results(name)
            rows = eval_observe.read_ledger(name)
    return rows[pending["ledger"]] if len(rows) > pending["ledger"] else None


def _run(name: str, state: dict, step: str, axis: str, note: str) -> dict:
    state["pending"] = state.get("pending") or {"step": step, "ledger": len(eval_observe.read_ledger(name))}
    _save(name, state)
    return ok(with_next({"proxy_name": name, "phase": state["phase"], "note": note},
                        f"proxy_eval(op='run', axis={axis!r}, confirm=True), then proxy_merge(op='next', confirm=True)"))


def _restore(name: str, tree: str) -> None:
    cfg = load_opt_config(name) or {}
    tree_snapshot.restore(opt_git_dir(), workspace_root(), list(cfg.get("optimize_paths") or []), tree)


def _finish(name: str, state: dict) -> dict:
    """Keep whichever was better, the whole or the chain, on disk and in the state."""
    cfg = load_opt_config(name) or {}
    goal = cfg.get("primary_goal", "min")
    whole = state.get("all") or {}
    tip = state.get("tip") or {}
    best = {"from": "baseline", "tree": cfg.get("baseline_id", ""), "value": state.get("base_value"),
            "kept": []}
    if whole.get("feasible") and is_improvement(whole.get("value"), best["value"], goal):
        best = {"from": "all", "tree": whole["tree"], "value": whole["value"], "kept": whole["applied"]}
    kept = [s["proxy"] for s in state.get("steps", []) if s.get("kept")]
    if tip.get("tree") and kept and is_improvement(tip.get("value"), best["value"], goal):
        best = {"from": "chain", "tree": tip["tree"], "value": tip["value"], "kept": kept}
    best["relative"] = relative(best["value"], state.get("base_value"), goal)
    if best["tree"]:
        _restore(name, best["tree"])
    state.update(phase="done", pending=None, final=best)
    _save(name, state)
    return report(name)


def next_(name: str = "") -> dict:
    name = resolve_proxy_name(name) or ""
    state = _load(name) if name else {}
    if not state:
        return err("No merge session.", hint="proxy_merge(op='start', proxies=[...], confirm=True) first.")
    cfg = load_opt_config(name) or {}
    goal = cfg.get("primary_goal", "min")
    margin, _ = effective_min_improvement(cfg)
    by_name = {s["proxy"]: s for s in state["sources"]}
    while True:
        pending = state.get("pending")
        row = _settled(name, pending) if pending else None
        if row == "running":
            return ok(with_next({"proxy_name": name, "phase": state["phase"],
                                 "note": "A run is under way; its verdict decides the next step."},
                                "wait for the run, then proxy_merge(op='next', confirm=True)"))
        phase = state["phase"]

        if phase == "baseline":
            cfg = load_opt_config(name) or {}
            if cfg.get("baseline_run_id"):
                rows = {r.get("run_id"): r for r in eval_observe.read_ledger(name)}
                state.update(base_value=(rows.get(cfg["baseline_run_id"]) or {}).get("primary_value"),
                             phase="all", pending=None)
                continue
            if row:
                return err("The baseline run of this copy did not pass: nothing can be compared to it.",
                           hint="proxy_eval_status(op='results') says why.")
            return _run(name, state, "baseline", "merge: baseline",
                        "Measuring this copy untouched first: every source is judged against it.")

        if phase == "all":
            if not pending:
                applied, conflicts = [], []
                for s in state["sources"]:
                    clash = apply(s)
                    (conflicts.append({"proxy": s["proxy"], "files": clash}) if clash
                     else applied.append(s["proxy"]))
                if not applied:
                    return _finish(name, state)
                state["all"] = {"applied": applied, "conflicts": conflicts,
                                "expected": _expected([by_name[p] for p in applied])}
                return _run(name, state, "all", "merge: all (" + " + ".join(applied) + ")",
                            f"{len(applied)} source(s) applied together"
                            + (f"; {len(conflicts)} set aside for a conflict" if conflicts else "") + ".")
            if not row:
                return _run(name, state, "all", "merge: all (" + " + ".join(state["all"]["applied"]) + ")",
                            "The sources are applied; the run that measures them is still to launch.")
            whole = state["all"]
            whole.update(run_id=row.get("run_id", ""), value=row.get("primary_value"),
                         feasible=bool(row.get("feasible")), tree=row.get("launch_tree", ""),
                         relative=relative(row.get("primary_value"), state["base_value"], goal))
            state["pending"] = None
            holds = whole["feasible"] and whole["relative"] is not None and \
                whole["relative"] <= whole["expected"] * (1 + margin)
            whole["holds"] = bool(holds)
            if holds or len(whole["applied"]) < 2:
                return _finish(name, state)
            _restore(name, cfg.get("baseline_id", ""))
            state.update(phase="incremental",
                         tip={"tree": cfg.get("baseline_id", ""), "value": state["base_value"]})
            continue

        if phase == "incremental":
            if pending and row:
                tip = state["tip"]
                src = by_name[pending["step"]]
                kept = bool(row.get("feasible")) and is_improvement(row.get("primary_value"), tip["value"], goal, margin)
                marginal = relative(row.get("primary_value"), tip["value"], goal)
                state["steps"].append({
                    "proxy": src["proxy"], "run_id": row.get("run_id", ""), "value": row.get("primary_value"),
                    "kept": kept, "alone": src["relative"], "marginal": marginal,
                    "interaction": round(marginal / src["relative"], 6) if marginal else None,
                    **({} if kept else {"why": "interference" if row.get("feasible") else "infeasible"}),
                })
                if kept:
                    state["tip"] = {"tree": row.get("launch_tree", ""), "value": row.get("primary_value")}
                else:
                    _restore(name, tip["tree"])
                state["pending"] = None
            elif pending:
                return _run(name, state, pending["step"], f"merge: + {pending['step']}",
                            f"{pending['step']} is applied; the run that measures it is still to launch.")
            done = {s["proxy"] for s in state["steps"]}
            for s in state["sources"]:
                if s["proxy"] in done:
                    continue
                clash = apply(s)
                if clash:
                    state["steps"].append({"proxy": s["proxy"], "kept": False, "why": "conflict",
                                           "files": clash, "alone": s["relative"]})
                    continue
                return _run(name, state, s["proxy"], f"merge: + {s['proxy']}",
                            f"Measured together, the sources fell short of the gain they promised: "
                            f"now one at a time. {s['proxy']} is applied over what was kept.")
            return _finish(name, state)

        return report(name)


def report(name: str = "") -> dict:
    name = resolve_proxy_name(name) or ""
    state = _load(name) if name else {}
    if not state:
        return err("No merge session.", hint="proxy_merge(op='start', proxies=[...], confirm=True) first.")
    whole = state.get("all") or {}
    final = state.get("final") or {}
    payload = {
        "proxy_name": name,
        "phase": state["phase"],
        "branch": state.get("branch", ""),
        "sources": [_public(s) for s in state["sources"]],
        "pairs": state.get("pairs", []),
        "expected": {"relative": state["expected"], "gain": round(1 - state["expected"], 6)},
        "all": {k: whole.get(k) for k in ("applied", "conflicts", "expected", "relative", "holds", "run_id")
                if k in whole},
        "steps": state.get("steps", []),
    }
    if whole.get("relative") and whole.get("expected"):
        payload["all"]["interaction"] = round(whole["relative"] / whole["expected"], 6)
    if state["phase"] != "done":
        return ok(with_next(payload, "proxy_merge(op='next', confirm=True)"))
    kept = final.get("kept", [])
    payload.update(
        final={**{k: final.get(k) for k in ("from", "value", "relative")},
               "gain": round(1 - final["relative"], 6) if final.get("relative") else None},
        kept=kept,
        set_aside=[{"proxy": s["proxy"], "why": next(
            (st.get("why") for st in state.get("steps", []) if st["proxy"] == s["proxy"]),
            "conflict" if any(c["proxy"] == s["proxy"] for c in whole.get("conflicts", [])) else "")}
            for s in state["sources"] if s["proxy"] not in kept],
        note=("An interaction under 1 means a source did better in the whole than alone, over 1 "
              "that the whole took some of its gain. The files are left as the better of the whole "
              "and the one-at-a-time chain; they are committed to the branch when you finish."),
    )
    return ok(with_next(payload, "answer with this report; the merged code is on " + (state.get("branch") or "this copy's branch")))
