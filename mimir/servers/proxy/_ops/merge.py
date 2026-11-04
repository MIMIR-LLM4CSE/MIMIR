"""proxy_merge ops: plan, start, next, report. The proxies' kept patches (``merge_sources``)
merged in one copy and measured there: all at once first, then one by one when the whole
falls short of what the parts promised. The merge is an ordinary proxy session,
``merge-<stamp>``, whose ``merge.json`` holds the sources, the phase and what each did.
"""
from __future__ import annotations

import copy
import os
import time
from datetime import datetime, timezone

from . import check_name, err, ok, with_next
from ....shared.atomic import write_json
from ....shared.state_paths import SUBAGENTS_DIRNAME, active_session_id
from .._lib import tree_snapshot
from .._lib.layout import merge_file, read_json, resolve_proxy_name
from .._lib.procs import opt_active_run_dir, run_state
from .._lib.ratchet import effective_min_improvement, is_improvement, relative
from .._lib.store import (load_opt_config, load_registry, opt_git_dir, registry_lock,
                          save_opt_config, save_registry, workspace_root)
from . import eval_observe, eval_session, merge_sources

_TOOLS = ["proxy_merge", "proxy_eval", "proxy_eval_status"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    sources, refused, refusal = merge_sources.gather(names)
    if refusal:
        return err(refusal, refused=refused)
    picked = [s["proxy"] for s in sources]
    expected = merge_sources.expected(sources)
    return ok(with_next({
        "sources": [merge_sources.public(s) for s in sources],
        "refused": refused,
        "pairs": merge_sources.pairs(sources),
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
    """Whether this server works for a sub-agent: one that may run a tool works in a copy
    of its own."""
    return f"/{SUBAGENTS_DIRNAME}/" in "/" + active_session_id()


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


def _moved(value, src_root: str, here: str):
    """*value* with every path under *src_root* moved under *here*, wherever it is in it
    (a command template names several)."""
    if isinstance(value, str):
        return here if value == src_root else value.replace(src_root + os.sep, here + os.sep)
    if isinstance(value, list):
        return [_moved(v, src_root, here) for v in value]
    if isinstance(value, dict):
        return {k: _moved(v, src_root, here) for k, v in value.items()}
    return value


def start(names: list[str], name: str = "") -> dict:
    if not _in_copy():
        return err("A merge is measured in a copy of its own, and this is the main workspace.",
                   hint="proxy_merge(op='plan', proxies=[...]) gives the brief to hand to spawn_agent.")
    sources, refused, refusal = merge_sources.gather(names)
    if refusal:
        return err(refusal, refused=refused)
    name = name or f"merge-{time.strftime('%Y%m%d-%H%M%S')}"
    bad = check_name("name", name)
    if bad:
        return bad
    here = workspace_root()
    configs = {s["proxy"]: load_opt_config(s["proxy"]) or {} for s in sources}
    roots = {p: _root_of(c) for p, c in configs.items()}
    lost = [p for p, r in roots.items() if not r]
    if lost:
        return err(f"Where the files of {', '.join(lost)} lived cannot be told, so their paths cannot be moved here.")
    first = sources[0]
    cfg, root = configs[first["proxy"]], roots[first["proxy"]]
    entry = (load_registry() or {}).get(first["proxy"])
    if not entry:
        return err(f"{first['proxy']} is no longer registered: its harness cannot be found.")
    harness = _moved(cfg.get("proxy_source_path", ""), root, here)
    if not os.path.isfile(harness):
        return err(f"The harness {os.path.relpath(harness, here)} is not in this copy: it was "
                   f"committed only on {first['branch'] or 'its own copy'}.",
                   hint=f"git checkout {first['branch'] or '<branch>'} -- {os.path.relpath(harness, here)}, "
                        "then proxy_merge(op='start') again.")
    paths = sorted({_moved(path, roots[p], here) for p, c in configs.items() for path in c.get("optimize_paths") or []})
    with registry_lock(create=True):
        reg = load_registry() or {}
        if name in reg:
            return err(f"A proxy named {name} is already registered.")
        reg[name] = {**_moved(copy.deepcopy(entry), root, here), "name": name,
                     "registered_at": _now(), "merged_from": [s["proxy"] for s in sources]}
        reg[name].pop("updated_at", None)
        save_registry(reg)
    res = eval_session.init(
        name, first["suite"], cfg.get("requirements") or [], harness, paths,
        _moved(cfg.get("python_executable", ""), root, here), cfg.get("max_hours", 0.0),
        first["metric"], first["goal"], cfg.get("min_improvement", 0.02),
        # Out of reach: a merge runs once all together and once per source, and a
        # "converged" verdict would say nothing about it.
        max_stall=len(sources) + 2, convergence=cfg.get("convergence"), repeat=cfg.get("repeat", 0))
    if res.get("status") != "ok":
        return res
    mine = load_opt_config(name)
    mine["merge"] = True
    save_opt_config(mine)
    expected = merge_sources.expected(sources)
    _save(name, {
        "name": name, "branch": mine.get("branch", ""), "started_at": _now(),
        "sources": sources, "refused": refused, "pairs": merge_sources.pairs(sources),
        "expected": expected, "phase": "baseline", "pending": None,
        "base_value": None, "all": None, "steps": [], "final": None,
    })
    return ok(with_next({"proxy_name": name, "sources": [merge_sources.public(s) for s in sources],
                         "refused": refused, "expected": expected},
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


def _why(state: dict, proxy: str, kept_from: str) -> str:
    """Why *proxy* is not in the code kept, which came *kept_from* the whole, the chain or
    neither: what set it aside there, a conflict, or a whole that gained nothing (or failed)."""
    whole = state.get("all") or {}
    clash = any(c["proxy"] == proxy for c in whole.get("conflicts", []))
    if kept_from == "all":
        return "conflict"
    step = next((s for s in reversed(state.get("steps", [])) if s["proxy"] == proxy), None)
    if step and not step["kept"]:
        return step["why"]
    if clash:
        return "conflict"
    return "infeasible" if whole and not whole.get("feasible") else "no_gain"


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
    best["set_aside"] = [{"proxy": s["proxy"], "why": _why(state, s["proxy"], best["from"])}
                         for s in state["sources"] if s["proxy"] not in best["kept"]]
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
                    clash = merge_sources.apply(s)
                    (conflicts.append({"proxy": s["proxy"], "files": clash}) if clash
                     else applied.append(s["proxy"]))
                if not applied:
                    return _finish(name, state)
                state["all"] = {"applied": applied, "conflicts": conflicts,
                                "expected": merge_sources.expected([by_name[p] for p in applied])}
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
                clash = merge_sources.apply(s)
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
        "sources": [merge_sources.public(s) for s in state["sources"]],
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
        set_aside=final.get("set_aside", []),
        note=("An interaction under 1 means a source did better in the whole than alone, over 1 "
              "that the whole took some of its gain. The files are left as the better of the whole "
              "and the one-at-a-time chain; they are committed to the branch when you finish."),
    )
    return ok(with_next(payload, "answer with this report; the merged code is on " + (state.get("branch") or "this copy's branch")))
