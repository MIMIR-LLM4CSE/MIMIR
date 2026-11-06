"""proxy_merge ops: plan, start, next, resolved, pause, drop, report.

Each proxy's kept patch is picked as a git commit on a branch of the merge's own: in the
workspace, which goes back to its branch once the merge is measured or paused, or in a
sub-agent's copy (``merge_sources.kept_commit``); a conflict stops the merge for the
agent, or the user, to resolve. The merged code is then the proxy ``merge-<stamp>``, over
every source's cases. ``merge.json`` holds the sources, the phase and what each did.
"""
from __future__ import annotations

import copy
import os
import re
import time
from datetime import datetime, timezone

from . import check_name, err, ok, with_next
from ....shared.atomic import read_json, write_json
from ....shared.state_paths import SUBAGENTS_DIRNAME, active_session_id
from .._lib.layout import (cache_dir, merge_file, opt_runs_dir, opt_session_runs_dir, registry_path,
                           resolve_proxy_name)
from .._lib.procs import opt_active_run_dir, run_state
from .._lib.ratchet import (baseline_value, effective_min_improvement, load_best, read_ledger,
                            relative, run_case_times)
from .._lib.store import (clear_active_session, load_opt_config, load_registry, load_suite, registry_lock,
                          save_opt_config, save_registry, save_suite, suite_path, workspace_root,
                          write_active_session)
from . import eval_observe, eval_session, merge_sources

_TOOLS = ["proxy_merge", "proxy_eval", "proxy_eval_status", "read_file_lines", "replace_in_file",
          "write_file", "ask_user_question"]

# The lines shown around each conflict, and the most conflicts shown per file.
_CONTEXT = 3
_MAX_HUNKS = 6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _brief(names: list[str]) -> str:
    return (
        f"Merge the kept optimisations of the proxies {', '.join(names)} and measure the whole. In "
        f"your copy: proxy_merge(op='start', proxies={names!r}, confirm=True), then follow each "
        "next_step: proxy_merge(op='next', confirm=True) and the proxy_eval(op='run', ...) it names. "
        "When a proxy's patch conflicts, next shows both sides and what each proxy was doing. A "
        "trivial conflict (an import, neighbouring lines, both keeping the same intent) you resolve "
        "yourself in the files, then proxy_merge(op='resolved', resolution=<how>, confirm=True). A "
        "conflict of method (two rewrites of the same loop, two layouts of the same data: one or "
        "the other) or any doubt: reason about it, then ask the user with ask_user_question, giving "
        "both options and your recommendation. If no answer comes, proxy_merge(op='pause', "
        "question=<it>, confirm=True) and stop with the question. When the whole is measured, a "
        "kernel that lost its gain in the merge is a question for the user too (keep it, drop that "
        "proxy with proxy_merge(op='drop', proxies=[...], confirm=True), or rework the axis). "
        "Answer with proxy_merge(op='report')."
    )


def warnings(pairs: list[dict]) -> list[str]:
    """What the user is to know before trusting a merge: proxies that ran at the same time
    in one tree, on a file both change."""
    return [f"{p['a']} and {p['b']} ran at the same time in the same tree and both change "
            f"{', '.join(p['shared'])}: each one's kept patch, and its gain alone, may hold the other's "
            "work. A conflict between them may be with the other's state of the moment, and their "
            "kernels' gain in the merge against alone says less. Tell the user before merging."
            for p in pairs if p.get("together")]


def plan(names: list[str]) -> dict:
    sources, refused, refusal = merge_sources.gather(names)
    if refusal:
        return err(refusal, refused=refused)
    picked = [s["proxy"] for s in sources]
    pairs = merge_sources.pairs(sources)
    return ok(with_next({
        "sources": [merge_sources.public(s) for s in sources],
        "refused": refused,
        "pairs": pairs,
        "warnings": warnings(pairs),
        "order": picked,
        "brief": _brief(picked),
        "note": ("Each proxy's kept patch is merged with git, one after the other, best alone first, on "
                 "a branch of the merge's own. Conflicts are resolved there: the trivial ones by you, a "
                 "choice of method by the user. The merged code is then measured on every proxy's own "
                 "cases: each kernel against its gain alone, the whole as their total time."),
    }, f"proxy_merge(op='start', proxies={picked!r}, confirm=True), yourself: the workspace goes on the "
       "merge's branch and back on its own once the merge is measured. Several combinations worth "
       "comparing: one plan each, and spawn_agent(task=<its brief>, "
       f"tools={_TOOLS!r}, background=True) per combination, all in one response"))


def _in_copy() -> bool:
    """Whether this server works for a sub-agent: one that may run a tool works in a copy
    of its own, already on a branch of its own."""
    return f"/{SUBAGENTS_DIRNAME}/" in "/" + active_session_id()


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


def entry_name(merge: str, proxy: str) -> str:
    """The registry entry that runs *proxy*'s cases in the merge *merge*."""
    return f"{merge}--{proxy}"


def _case_id(proxy: str, case_id: str) -> str:
    return f"{proxy}/{case_id}"


# ── the workspace on the merge's branch ───────────────────────────────────────

def _runs_here() -> list[str]:
    """The sessions of this workspace with a run under way or queued: it measures the
    files on disk, which a switch of branch changes under it."""
    here = workspace_root()
    try:
        names = sorted(os.listdir(opt_runs_dir()))
    except OSError:
        return []
    busy = []
    for n in names:
        cfg = load_opt_config(n)
        if not cfg or (cfg.get("root") or here) != here:
            continue
        active = opt_active_run_dir(n)
        if active and run_state(active)["state"] in ("running", "pending"):
            busy.append(n)
    return busy


def _switch_refusal() -> dict | None:
    """Why the workspace cannot change branch now: changes git tracks, or a run on its files."""
    changed = merge_sources.dirty()
    if changed:
        more = "…" if len(changed) > 5 else ""
        return err(f"The workspace has changes git tracks ({', '.join(changed[:5])}{more}): the merge's "
                   "branch cannot be checked out over them.",
                   hint="The kept patches come from the snapshot store, not from these files: ask the user "
                        "whether to commit them or to git stash them, then call proxy_merge again.")
    busy = _runs_here()
    if busy:
        return err(f"{', '.join(busy)} has a run under way or queued on this workspace's files: a switch of "
                   "branch would change the code under it.",
                   hint="Wait for it (proxy_eval_status(op='status')), then call proxy_merge again.")
    return None


def holder() -> tuple[str, dict]:
    """The merge whose branch this workspace is on, with its state; ("", {}) when none.
    While one holds it, the files on disk are the merged code."""
    here = workspace_root()
    try:
        names = sorted(os.listdir(opt_runs_dir()))
    except OSError:
        return "", {}
    merges = [(n, st) for n, st in ((n, _load(n)) for n in names)
              if st.get("in_workspace") and st.get("root") == here]
    if not merges:
        return "", {}
    on = merge_sources.current_ref()
    return next(((n, st) for n, st in merges if st.get("branch") == on), ("", {}))


def held_against(proxy: str) -> dict | None:
    """The refusal of an op that runs or writes *proxy*'s files while a merge holds the
    workspace; None when none does, or *proxy* is that merge."""
    name, state = holder()
    if not name or (resolve_proxy_name(proxy) or "") == name:
        return None
    return err(f"The workspace is on {state['branch']}, the merge {name}'s branch: its files are the merged "
               f"code, not {proxy or 'this session'}'s.",
               hint=f"Carry the merge on (proxy_merge(op='next', proxy_name={name!r}, confirm=True)) until it "
                    f"is measured, or proxy_merge(op='pause', proxy_name={name!r}, question=<it>, confirm=True): "
                    f"either takes the workspace back to {state.get('return_to') or 'its branch'}.")


def _enter(name: str, state: dict) -> dict | None:
    """The workspace on the merge's branch and its session the active one, or the refusal
    when it cannot switch. Nothing to do in a copy."""
    if not state.get("in_workspace"):
        return None
    on = merge_sources.current_ref()
    if on != state["branch"]:
        refusal = _switch_refusal()
        if refusal:
            return refusal
        out = merge_sources.git("checkout", state["branch"])
        if out.returncode != 0:
            return err(f"{state['branch']} cannot be checked out: {out.stderr.strip()[-300:]}")
        state["return_to"] = on
    active = resolve_proxy_name("") or ""
    if active != name:
        state["was_active"] = active
        write_active_session(name)
    _save(name, state)
    return None


def _leave(name: str, state: dict) -> None:
    """The workspace back on the branch it was on before the merge, and the session that
    was active then back too. Nothing to do in a copy, or when it is elsewhere already."""
    if not state.get("in_workspace") or merge_sources.current_ref() != state["branch"]:
        return
    out = merge_sources.git("checkout", state.get("return_to") or "")
    if out.returncode != 0:
        state["stuck"] = out.stderr.strip()[-300:]
        return
    state.pop("stuck", None)
    if (resolve_proxy_name("") or "") == name:
        if state.get("was_active"):
            write_active_session(state["was_active"])
        else:
            clear_active_session()


def _workspace(state: dict) -> dict:
    """Where the workspace is, for an answer: on the merge's branch or back on its own."""
    if not state.get("in_workspace"):
        return {}
    on = merge_sources.current_ref()
    out = {"on": on, "merge_branch": state["branch"], "return_to": state.get("return_to", "")}
    if state.get("stuck") and on == state["branch"]:
        out["stuck"] = (f"{state.get('return_to')} could not be checked out back: {state['stuck']}. "
                        "Tell the user; their workspace is still on the merge's branch.")
    return out


def _undo(name: str, entries: list[str], head: str = "", back: str = "") -> None:
    """What ``start`` wrote, removed: the merge's entries, its suite, the commits it made
    since *head*, and in the workspace its branch, *back* checked out again."""
    if head:
        merge_sources.git("reset", "--keep", head)
    if back:
        branch = merge_sources.current_ref()
        merge_sources.git("checkout", back)
        merge_sources.git("branch", "-D", branch)
    with registry_lock(create=True):
        reg = load_registry() or {}
        for e in [name, *entries]:
            reg.pop(e, None)
        save_registry(reg)
    try:
        os.remove(suite_path(name))
    except OSError:
        pass


def _free_name(reg: dict, stamp: str) -> str:
    """``merge-<stamp>``, or with ``-2``, ``-3``… when another merge took it in the same
    second (several started at once, each in its sub-agent's copy)."""
    n = 1
    while True:
        name = f"merge-{stamp}" + (f"-{n}" if n > 1 else "")
        if not any(k == name or k.startswith(name + "--") for k in reg):
            return name
        n += 1


def start(names: list[str], name: str = "") -> dict:
    sources, refused, refusal = merge_sources.gather(names)
    if refusal:
        return err(refusal, refused=refused)
    if name:
        bad = check_name("name", name)
        if bad:
            return bad
    in_workspace = not _in_copy()
    if in_workspace:
        if merge_sources.tracked(opt_runs_dir(), registry_path()):
            return err(f"git tracks the proxy store {os.path.relpath(cache_dir(), workspace_root())}: a switch "
                       "to the merge's branch would swap its runs and its registry along with the code.",
                       hint="Ask the user to add it to .gitignore and untrack it (git rm -r --cached), "
                            "then call proxy_merge again.")
        held, _ = holder()
        if held:
            return err(f"The workspace is on the branch of the merge {held}.",
                       hint=f"Carry it on or pause it (proxy_merge(op='next'|'pause', proxy_name='{held}', "
                            "confirm=True)), then start this one.")
        blocked = _switch_refusal()
        if blocked:
            return blocked
    here = workspace_root()
    configs = {s["proxy"]: load_opt_config(s["proxy"]) or {} for s in sources}
    roots = {p: merge_sources.root_of(c) for p, c in configs.items()}
    lost = [p for p, r in roots.items() if not r]
    if lost:
        return err(f"Where the files of {', '.join(lost)} lived cannot be told, so their paths cannot be moved here.")
    reg = load_registry() or {}
    gone = [s["proxy"] for s in sources if s["proxy"] not in reg]
    if gone:
        return err(f"{', '.join(gone)} no longer registered: their harness cannot be found.")
    first = sources[0]
    cfg, root = configs[first["proxy"]], roots[first["proxy"]]
    harness = _moved(cfg.get("proxy_source_path", ""), root, here)
    if not os.path.isfile(harness):
        return err(f"The harness {os.path.relpath(harness, here)} is not in this tree: it was "
                   f"committed only on {first['branch'] or 'its own copy'}.",
                   hint=f"git checkout {first['branch'] or '<branch>'} -- {os.path.relpath(harness, here)}, "
                        "then proxy_merge(op='start') again.")
    suites = {s["proxy"]: load_suite(s["suite"]) for s in sources}
    lost = [f"{p}'s suite {s['suite']}" for s in sources for p in [s["proxy"]] if not suites[p]]
    if lost:
        return err(f"{', '.join(lost)} is gone: its cases cannot be measured.")
    paths = sorted({_moved(path, roots[p], here) for p, c in configs.items() for path in c.get("optimize_paths") or []})
    with registry_lock(create=True):
        reg = load_registry() or {}
        name = name or _free_name(reg, time.strftime("%Y%m%d-%H%M%S"))
        entries = [entry_name(name, s["proxy"]) for s in sources]
        taken = [e for e in [name, *entries] if e in reg]
        if taken:
            return err(f"{', '.join(taken)} already registered.")
        reg[name] = {**_moved(copy.deepcopy(reg[first["proxy"]]), root, here), "name": name,
                     "registered_at": _now(), "merged_from": [s["proxy"] for s in sources]}
        reg[name].pop("updated_at", None)
        for s, e in zip(sources, entries):
            reg[e] = {**_moved(copy.deepcopy(reg[s["proxy"]]), roots[s["proxy"]], here), "name": e,
                      "registered_at": _now(), "merge_of": name}
            reg[e].pop("updated_at", None)
        save_registry(reg)
    # Every source's cases, each run by its source's harness (its paths in this tree) and judged by its
    # source's requirements.
    cases = []
    for s in sources:
        for case in (suites[s["proxy"]] or {}).get("cases") or []:
            c = _moved(copy.deepcopy(case), roots[s["proxy"]], here)
            cases.append({**c, "case_id": _case_id(s["proxy"], case.get("case_id", "")),
                          "proxy_name": entry_name(name, s["proxy"]),
                          "requirements": c.get("requirements", s["requirements"]),
                          "source": s["proxy"]})
    save_suite(name, {"name": name, "created_at": _now(), "cases": cases,
                      "description": "The cases of " + ", ".join(s["proxy"] for s in sources) + ", merged"})
    back, was_active = "", ""
    if in_workspace:
        back, was_active = merge_sources.current_ref(), resolve_proxy_name("") or ""
        out = merge_sources.git("checkout", "-b", f"mimir/{name}")
        if out.returncode != 0:
            _undo(name, entries)
            return err(f"The merge's branch mimir/{name} cannot be made: {out.stderr.strip()[-300:]}")
    head = merge_sources.git("rev-parse", "HEAD").stdout.strip()
    # A proxy's work already committed on this branch is taken out before the tree is
    # measured untouched: else that untouched tree holds it, and its kernel would seem to
    # gain nothing in the merge. The merge picks it back like the others.
    unapplied = []
    for s in sources:
        if not merge_sources.already_in(s):
            continue
        sha, why = merge_sources.kept_commit(s)
        taken_out = merge_sources.git("revert", "--no-edit", sha) if sha else None
        if taken_out is None or taken_out.returncode != 0:
            merge_sources.git("revert", "--abort")
            _undo(name, entries, head, back)
            return err(f"{s['proxy']}'s work is already on this branch and cannot be taken out to "
                       f"measure the code without it: {why or (taken_out.stderr.strip()[-300:] if taken_out else '')}")
        unapplied.append(s["proxy"])
    res = eval_session.init(
        name, name, [], harness, paths,
        _moved(cfg.get("python_executable", ""), root, here), cfg.get("max_hours", 0.0),
        "total_time_s", "min", cfg.get("min_improvement", 0.02),
        # Out of reach: a merge runs its baseline, the whole and again after a drop, and a
        # "converged" verdict would say nothing about it.
        max_stall=len(sources) + 3, convergence=None, repeat=cfg.get("repeat", 0))
    if res.get("status") != "ok":
        _undo(name, entries, head, back)
        return res
    mine = load_opt_config(name)
    mine["merge"] = True
    save_opt_config(mine)
    state = {
        "name": name, "branch": mine.get("branch", ""), "started_at": _now(),
        "in_workspace": in_workspace, "root": here, "return_to": back, "was_active": was_active,
        "sources": sources, "refused": refused, "pairs": merge_sources.pairs(sources),
        "entries": entries, "suite": name, "phase": "baseline", "pending": None, "unapplied": unapplied,
        "merged": [], "conflict": None, "question": None, "dropped": [],
        "kernels": None, "whole": None,
    }
    _save(name, state)
    note = (f"The workspace is on {state['branch']} until the merge is measured or paused, then back on "
            f"{back}. Tell the user: their open files show the merged code meanwhile."
            if in_workspace else "")
    return ok(with_next({"proxy_name": name, "branch": state["branch"],
                         "sources": [merge_sources.public(s) for s in sources],
                         "refused": refused, "cases": [c["case_id"] for c in cases],
                         "unapplied": unapplied, **({"note": note} if note else {})},
                        "proxy_merge(op='next', confirm=True)"))


def _load(name: str) -> dict:
    return read_json(merge_file(name), {}) or {}


def _save(name: str, state: dict) -> None:
    write_json(merge_file(name), state, indent=2)


def _session(name: str) -> tuple[str, dict]:
    name = resolve_proxy_name(name) or ""
    return name, (_load(name) if name else {})


_NO_SESSION = ("No merge session.", "proxy_merge(op='start', proxies=[...], confirm=True) first.")


def _running(name: str, pending: dict) -> bool:
    """Whether the run *pending* waits for is still under way."""
    if len(read_ledger(name)) > pending["ledger"]:
        return False
    active = opt_active_run_dir(name)
    if not active:
        return False
    return run_state(active)["state"] in ("running", "pending")


def _settled(name: str, pending: dict) -> dict | None:
    """The run settled since *pending* was set, None when none was launched."""
    rows = read_ledger(name)
    if len(rows) <= pending["ledger"] and opt_active_run_dir(name):
        # Ended, not settled yet: settling is the results read's, whoever comes first.
        eval_observe.results(name)
        rows = read_ledger(name)
    return rows[pending["ledger"]] if len(rows) > pending["ledger"] else None


def _run(name: str, state: dict, step: str, axis: str, note: str) -> dict:
    state["pending"] = state.get("pending") or {"step": step, "ledger": len(read_ledger(name))}
    _save(name, state)
    return ok(with_next({"proxy_name": name, "phase": state["phase"], "note": note},
                        f"proxy_eval(op='run', axis={axis!r}, confirm=True), then proxy_merge(op='next', confirm=True)"))


def _baseline(name: str, state: dict, row: dict | None) -> dict | None:
    """The tree measured untouched first: every kernel is judged against it."""
    cfg = load_opt_config(name) or {}
    if cfg.get("baseline_run_id"):
        state.update(phase="merging", pending=None)
        return None
    if row:
        state["pending"] = None
        _save(name, state)
        return err("The baseline run of the untouched tree did not pass: nothing can be compared to it.",
                   hint="proxy_eval_status(op='results') says why; proxy_merge(op='next', confirm=True) measures it again.")
    return _run(name, state, "baseline", "merge: baseline",
                "Measuring the tree untouched first: every kernel is judged against it.")


def _axes(proxy: str) -> list[str]:
    """What *proxy* did, as the axes of its accepted runs say it."""
    return [r["axis"] for r in read_ledger(proxy) if r.get("verdict") == "accept" and r.get("axis")
            and not r.get("baseline")]


def _hunks(path: str) -> list[str]:
    """The conflicts git left in *path*, each with a few lines around it."""
    try:
        with open(os.path.join(workspace_root(), path), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    out: list[str] = []
    i = 0
    while i < len(lines) and len(out) < _MAX_HUNKS:
        if lines[i].startswith("<<<<<<< "):
            end = next((j for j in range(i, len(lines)) if lines[j].startswith(">>>>>>> ")), len(lines) - 1)
            lo, hi = max(0, i - _CONTEXT), min(len(lines), end + 1 + _CONTEXT)
            out.append("\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi)))
            i = end + 1
        else:
            i += 1
    return out


def _unmerged() -> list[str]:
    r = merge_sources.git("diff", "--name-only", "--diff-filter=U")
    return [p for p in r.stdout.splitlines() if p] if r.returncode == 0 else []


def _with_markers(paths: list[str]) -> list[str]:
    marker = re.compile(r"^(<{7}|>{7}) ", re.M)
    left = []
    for p in paths:
        try:
            with open(os.path.join(workspace_root(), p), encoding="utf-8", errors="replace") as fh:
                if marker.search(fh.read()):
                    left.append(p)
        except OSError:
            pass
    return left


def _conflict_answer(name: str, state: dict) -> dict:
    """The conflict the merge stopped on: each file's hunks, and what each side was doing."""
    c = state["conflict"]
    ours = [m["proxy"] for m in state["merged"] if m["proxy"] not in state["dropped"]]
    return ok(with_next({
        "proxy_name": name, "phase": state["phase"],
        "conflict": {
            "proxy": c["proxy"],
            "files": [{"path": p, "hunks": _hunks(p)} for p in c["files"]],
            # Between <<<<<<< and ======= what is merged so far, after it the proxy's.
            "ours": {"proxies": ours, "axes": {p: _axes(p) for p in ours}},
            "theirs": {"proxy": c["proxy"], "axes": _axes(c["proxy"])},
        },
        "note": ("Trivial (an import, neighbouring lines, the same intent on both sides): resolve it "
                 "in the files. A choice of method (one rewrite or the other) or any doubt: reason "
                 "about it and ask the user with ask_user_question, both options and your "
                 "recommendation, before you resolve."),
    }, "edit the files until no conflict marker is left, then "
       "proxy_merge(op='resolved', resolution=<how and why>, confirm=True); no answer from the user: "
       "proxy_merge(op='pause', question=<the question>, confirm=True) and stop with it"))


def _merging(name: str, state: dict, row: dict | None) -> dict | None:
    """Each source's kept commit picked over what is merged so far, best alone first."""
    if state.get("conflict"):
        return _conflict_answer(name, state)
    done = {m["proxy"] for m in state["merged"]}
    for s in state["sources"]:
        if s["proxy"] in done:
            continue
        sha, why = merge_sources.kept_commit(s)
        if not sha:
            return err(f"{s['proxy']}'s kept state cannot be built here: {why}.")
        # --keep-redundant-commits: a pick whose lines HEAD already holds is a success
        # that changes nothing, not a failure.
        picked = merge_sources.git("cherry-pick", "--keep-redundant-commits", sha)
        if picked.returncode == 0:
            head = merge_sources.git("rev-parse", "HEAD").stdout.strip()
            state["merged"].append({"proxy": s["proxy"], "commit": head, "conflicts": [], "resolution": ""})
            _save(name, state)
            continue
        files = _unmerged()
        if not files:
            merge_sources.git("cherry-pick", "--abort")
            return err(f"{s['proxy']}'s kept state did not apply: {picked.stderr.strip()[-400:]}")
        state["conflict"] = {"proxy": s["proxy"], "files": files}
        _save(name, state)
        return _conflict_answer(name, state)
    state["phase"] = "measure"
    return None


def _kernels(name: str, state: dict, row: dict) -> list[dict]:
    """Each source's cases in the merged run, against the merge's baseline, beside the same
    case in that source's best run against its own baseline."""
    cfg = load_opt_config(name) or {}
    merged = run_case_times(os.path.join(opt_session_runs_dir(name), row.get("run_id", "")))
    base = run_case_times(os.path.join(opt_session_runs_dir(name), cfg.get("baseline_run_id", "")))
    margin = effective_min_improvement(cfg)[0]
    out = []
    for s in state["sources"]:
        if s["proxy"] in state["dropped"]:
            continue
        scfg = load_opt_config(s["proxy"]) or {}
        runs = opt_session_runs_dir(s["proxy"])
        best = run_case_times(os.path.join(runs, (load_best(s["proxy"]) or {}).get("run_id", "")))
        alone_base = run_case_times(os.path.join(runs, scfg.get("baseline_run_id", "")))
        for case in sorted(alone_base):
            cid = _case_id(s["proxy"], case)
            alone = relative(best.get(case), alone_base.get(case), "min")
            in_merge = relative(merged.get(cid), base.get(cid), "min")
            interaction = round(in_merge / alone, 6) if alone and in_merge else None
            out.append({"proxy": s["proxy"], "case": case, "alone": alone, "in_merge": in_merge,
                        "interaction": interaction,
                        "flagged": interaction is not None and interaction > 1 + margin})
    return out


def _measure(name: str, state: dict, row: dict | None) -> dict | None:
    """The merged code measured once: each kernel, and the whole as their total time."""
    if not row:
        kept = [m["proxy"] for m in state["merged"] if m["proxy"] not in state["dropped"]]
        return _run(name, state, "measure", "merge: " + " + ".join(kept),
                    "Every kept patch is merged: measuring the whole on every source's cases.")
    if row.get("verdict") == "incomparable":
        state["pending"] = None
        _save(name, state)
        return err(f"The merged code ran on {row.get('machine') or 'another machine'}, not where its baseline ran: "
                   "its times compare to nothing.",
                   hint="proxy_merge(op='next', confirm=True) names the run again: launch it where the baseline ran.")
    cfg = load_opt_config(name) or {}
    base = baseline_value(cfg, read_ledger(name))
    state["kernels"] = _kernels(name, state, row)
    state["whole"] = {"run_id": row.get("run_id", ""), "feasible": bool(row.get("feasible")),
                      "value": row.get("primary_value"), "base_value": base,
                      "relative": relative(row.get("primary_value"), base, "min")}
    state.update(phase="done", pending=None)
    _leave(name, state)
    _save(name, state)
    return report(name)


_PHASES = {"baseline": _baseline, "merging": _merging, "measure": _measure}


def next_(name: str = "") -> dict:
    name, state = _session(name)
    if not state:
        return err(_NO_SESSION[0], hint=_NO_SESSION[1])
    if not _PHASES.get(state.get("phase", "")) and state.get("phase") != "done":
        return err(f"{name} is a merge of an earlier kind; it cannot be carried on.",
                   hint="proxy_merge(op='plan', ...) and start a new one.")
    if state.get("phase") == "done":
        return report(name)
    refusal = _enter(name, state)
    if refusal:
        return refusal
    # Carried on after a pause: the user's answer is in the task that resumed it.
    state["question"] = None
    while True:
        pending = state.get("pending")
        if pending and _running(name, pending):
            return ok(with_next({"proxy_name": name, "phase": state["phase"],
                                 "note": "A run is under way; its verdict decides the next step."},
                                "wait for the run, then proxy_merge(op='next', confirm=True)"))
        phase = _PHASES.get(state["phase"])
        if not phase:
            return report(name)
        out = phase(name, state, _settled(name, pending) if pending else None)
        if out is not None:
            return out


def resolved(name: str = "", resolution: str = "") -> dict:
    """The conflict the merge stopped on, resolved in the files: committed, then on."""
    name, state = _session(name)
    if not state:
        return err(_NO_SESSION[0], hint=_NO_SESSION[1])
    c = state.get("conflict")
    if not c:
        return err("The merge is not stopped on a conflict.", hint="proxy_merge(op='next', confirm=True)")
    if state.get("in_workspace") and merge_sources.current_ref() != state["branch"]:
        return err(f"The workspace is not on {state['branch']}, where the conflict is.",
                   hint=f"git checkout {state['branch']}, then resolve it there.")
    if not resolution.strip():
        return err("Say how the conflict was resolved and why (resolution=...): the report and the map keep it.")
    left = _with_markers(c["files"])
    if left:
        return err(f"Conflict markers are left in {', '.join(left)}.",
                   hint="Resolve every <<<<<<< ... >>>>>>> block, then proxy_merge(op='resolved') again.")
    added = merge_sources.git("add", "--", *c["files"])
    # A commit, not --continue: a resolution that keeps what was merged leaves the pick
    # empty, which --continue refuses. Committing ends a one-commit pick all the same.
    done = merge_sources.git("commit", "--allow-empty", "--no-edit")
    if added.returncode != 0 or done.returncode != 0:
        return err(f"The resolution could not be committed: {(done.stderr or added.stderr).strip()[-400:]}")
    head = merge_sources.git("rev-parse", "HEAD").stdout.strip()
    state["merged"].append({"proxy": c["proxy"], "commit": head, "conflicts": c["files"],
                            "resolution": resolution.strip()})
    state["conflict"] = None
    _save(name, state)
    return next_(name)


def pause(name: str = "", question: str = "") -> dict:
    """Stop for an answer: a conflict pending is undone (the tree never stays mid-merge),
    and the question is kept for whoever carries the merge on."""
    name, state = _session(name)
    if not state:
        return err(_NO_SESSION[0], hint=_NO_SESSION[1])
    pending = state.get("pending")
    if pending and _running(name, pending):
        return err("A run of the merge is under way: the workspace stays on its branch until it ends.",
                   hint="Wait for it, then proxy_merge(op='next', confirm=True) or pause again.")
    c = state.get("conflict")
    if c:
        merge_sources.git("cherry-pick", "--abort")
    state["conflict"] = None
    state["question"] = {"proxy": c["proxy"] if c else "", "question": question.strip(), "at": _now()}
    _leave(name, state)
    _save(name, state)
    if state.get("in_workspace"):
        step = (f"stop here and ask the user the question; once answered, proxy_merge(op='next', "
                f"proxy_name={name!r}, confirm=True) takes the workspace back to {state['branch']} and "
                "carries the merge on")
    else:
        step = ("stop here and hand the question back; once answered, a new sub-agent carries the merge on "
                "with proxy_merge(op='next', confirm=True) in this copy's branch")
    return ok(with_next({"proxy_name": name, "phase": state["phase"], "question": state["question"],
                         "workspace": _workspace(state)}, step))


def drop(name: str = "", proxies: list[str] | None = None) -> dict:
    """Take *proxies* back out of the merged code (``git revert`` of their commits), then
    measure the whole again."""
    name, state = _session(name)
    if not state:
        return err(_NO_SESSION[0], hint=_NO_SESSION[1])
    if state.get("conflict") or state["phase"] in ("baseline", "merging"):
        return err("A proxy is dropped once the merge is measured.", hint="proxy_merge(op='next', confirm=True)")
    by_name = {m["proxy"]: m for m in state["merged"]}
    unknown = [p for p in proxies or [] if p not in by_name or p in state["dropped"]]
    if unknown or not proxies:
        return err(f"Not in the merged code: {', '.join(unknown) or '(none named)'}.")
    refusal = _enter(name, state)
    if refusal:
        return refusal
    for p in proxies:
        commit = by_name[p]["commit"]
        # A pick its resolution emptied changed nothing: there is nothing to take back.
        if merge_sources.git("diff", "--quiet", f"{commit}^", commit).returncode == 0:
            state["dropped"].append(p)
            continue
        r = merge_sources.git("revert", "--no-edit", commit)
        if r.returncode != 0:
            merge_sources.git("revert", "--abort")
            return err(f"{p} cannot be taken back out cleanly: {r.stderr.strip()[-400:]}",
                       hint="Another proxy's patch builds on its lines; ask the user what to keep.")
        state["dropped"].append(p)
    state.update(phase="measure", pending=None, kernels=None, whole=None)
    _save(name, state)
    return next_(name)


def report(name: str = "") -> dict:
    name, state = _session(name)
    if not state:
        return err(_NO_SESSION[0], hint=_NO_SESSION[1])
    kernels = state.get("kernels") or []
    whole = state.get("whole") or {}
    flagged = [k for k in kernels if k.get("flagged")]
    payload = {
        "proxy_name": name,
        "phase": state["phase"],
        "branch": state.get("branch", ""),
        "sources": [merge_sources.public(s) for s in state["sources"]],
        "merged": state.get("merged", []),
        "dropped": state.get("dropped", []),
        "conflict": state.get("conflict"),
        "question": state.get("question"),
        "kernels": kernels,
        "warnings": warnings(state.get("pairs", [])),
        "whole": {**whole, "gain": round(1 - whole["relative"], 6) if whole.get("relative") else None},
        **({"workspace": _workspace(state)} if state.get("in_workspace") else {}),
    }
    if state["phase"] != "done":
        return ok(with_next(payload, "proxy_merge(op='next', confirm=True)"))
    payload["note"] = (
        "Each kernel: its gain alone (alone), in the merge (in_merge), and their ratio "
        "(interaction: under 1 it did better merged, over 1 the merge took some of its gain). The "
        "whole is the total time of every case against the tree untouched. The merged code is "
        "committed on the branch; once the user has merged it, the source proxies can be cleaned "
        "(proxy_manage(op='clean')), their graph stays on the map: suggest it, never do it unasked.")
    back = (f"; the workspace is back on {state['return_to']}"
            if state.get("in_workspace") and merge_sources.current_ref() == state.get("return_to") else "")
    if flagged:
        names = sorted({k["proxy"] for k in flagged})
        return ok(with_next(payload,
                            f"{', '.join(names)} lost gain in the merge: reason about why, then ask the user "
                            "with ask_user_question whether to keep it, drop it (proxy_merge(op='drop', "
                            f"proxy_name={name!r}, proxies=[...], confirm=True)) or rework that axis on the "
                            "merged code" + back))
    return ok(with_next(payload, "answer with this report; the merged code is on "
                        + (state.get("branch") or "its branch") + back))
