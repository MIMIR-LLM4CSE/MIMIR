"""proxy_request ops: open, update, list, get. An R&D request from the code map — a
method, an algorithm, external code to bring in — on named targets, studied or tried.
Opening one records where the session stands, so its variants all start from there."""
from __future__ import annotations

from . import err, ok, with_next
from ....shared.state_paths import active_session_id
from .._lib import requests
from .._lib.layout import resolve_proxy_name
from .._lib.ratchet import load_best
from .._lib.store import load_opt_config
from .._lib.ratchet import read_ledger


def start_of(proxy_name: str) -> dict:
    """Where the session stands: its best accepted run's tree, else its baseline's; {}
    without a session."""
    cfg = load_opt_config(proxy_name)
    name = cfg.get("proxy_name", "") if cfg else ""
    if not name:
        return {}
    best = load_best(name) or {}
    if best.get("tree_snapshot"):
        run_id, tree = best.get("run_id", ""), best["tree_snapshot"]
    else:
        run_id, tree = cfg.get("baseline_run_id", ""), cfg.get("baseline_id", "")
    value = next((r.get("primary_value") for r in read_ledger(name) if r.get("run_id") == run_id), None)
    return {"proxy": name, "run_id": run_id, "tree": tree, "value": value,
            "metric": cfg.get("primary_metric", "time_s"), "goal": cfg.get("primary_goal", "min")}


def _clean_targets(targets: list | None) -> list[dict]:
    """The targets: a file with, when given, a function and its line; a whole module or
    component; or the whole code (``{"all": true}``)."""
    out = []
    for t in targets or []:
        if isinstance(t, str):
            t = {"file": t}
        if not isinstance(t, dict) or not (t.get("file") or t.get("module") or t.get("component") or t.get("all")):
            continue
        out.append({k: t[k] for k in ("file", "function", "line", "module", "component", "all") if t.get(k)})
    return out


def _try_steps(rid: str, variants: list[str]) -> str:
    """The calls that try each variant from the request's start."""
    n = max(1, len(variants))
    return (f"for each variant k of 1..{n}: proxy_eval(op='reset_to_request', request_id='{rid}', "
            f"confirm=True), implement the variant, then proxy_eval(op='run', axis='[{rid}.k] "
            "<the variant in a few words>', confirm=True); once all are measured, "
            f"proxy_eval(op='reset_to_best', confirm=True) and proxy_request(op='update', request_id='{rid}', "
            "status='done', conclusion=...)")


def open_(proxy_name: str, token: str, mode: str, kind: str, targets: list | None, idea: str,
          variants: list[str] | None, source: str, constraints: str) -> dict:
    if mode not in requests.MODES:
        return err(f"mode must be one of {', '.join(requests.MODES)}.")
    if kind not in requests.KINDS:
        return err(f"kind must be one of {', '.join(requests.KINDS)}.")
    if not idea.strip():
        return err("A request needs its idea: what to study or try.")
    if token and any(r.get("token") == token for r in requests.load()):
        found = next(r for r in requests.load() if r.get("token") == token)
        return ok(with_next({"request": found, "note": "Already opened from this message."},
                            "carry on with " + found["id"]))
    start = start_of(resolve_proxy_name(proxy_name) or "")
    kept = [v.strip() for v in variants or [] if v and v.strip()]
    row = requests.add({
        "token": token, "mode": mode, "kind": kind, "targets": _clean_targets(targets),
        "idea": idea.strip(), "variants": kept, "source": source.strip(),
        "constraints": constraints.strip(), "start": start,
        # The conversation the request was put in: its study is the agent's answer there,
        # and the map's link goes back to it.
        "session": active_session_id(),
        "status": "studying" if mode == "study" else "open",
    })
    rid = row["id"]
    # A study is delivered as the answer in the chat, where the map's link leads; the
    # request keeps only its status and a line.
    if mode == "study" and kind == "refactor":
        nxt = ("read the map's findings in the message and the targets' code; answer in the chat with the "
               "target architecture and the steps to reach it; then "
               f"proxy_request(op='update', request_id='{rid}', status='proposed', summary=<one line>). Edit no code.")
    elif mode == "study":
        nxt = ("read the targets and their profile; answer in the chat with the analysis and the plan; then "
               f"proxy_request(op='update', request_id='{rid}', status='proposed', summary=<one line>). Edit no code.")
    elif not start:
        nxt = ("no optimisation session is open: set one up as the proxy-optimize skill says, "
               f"then proxy_request(op='update', request_id='{rid}', status='trying') and " + _try_steps(rid, kept))
    else:
        nxt = f"proxy_request(op='update', request_id='{rid}', status='trying'), then " + _try_steps(rid, kept)
    return ok(with_next({"request": row, "axis": f"[{rid}.k] <variant>" if len(kept) > 1 else f"[{rid}] <what it tries>"},
                        nxt))


def update(request_id: str, status: str, summary: str, conclusion: str, run_id: str = "") -> dict:
    if status and status not in requests.STATUSES:
        return err(f"status must be one of {', '.join(requests.STATUSES)}.")
    changes: dict = {k: v.strip() for k, v in (("status", status), ("summary", summary), ("conclusion", conclusion)) if v}
    if run_id.strip():
        # A run launched without the request's axis tag, attached after the fact.
        known = (requests.get(request_id) or {}).get("runs") or []
        changes["runs"] = known + [run_id.strip()] if run_id.strip() not in known else known
    if not changes:
        return err("Nothing to update: give status, summary, conclusion or run_id.")
    row = requests.update(request_id, changes)
    if row is None:
        return err(f"No request {requests.normalise(request_id)}.",
                   hint="proxy_request(op='list') names them.")
    return ok({"request": row})


def untagged_hint(proxy_name: str, axis: str, run_id: str) -> dict:
    """For a run whose axis names no request while some are still open on this proxy: the
    open ones and the call that attaches the run, so a request continued from the chat
    rather than from the map still shows its runs. {} otherwise."""
    if requests.axis_of(axis):
        return {}
    open_ = [r for r in requests.load()
             if r.get("status") not in ("done", "dropped")
             and (r.get("start") or {}).get("proxy", proxy_name) in (proxy_name, "")]
    if not open_:
        return {}
    return {
        "open_requests": [{k: r.get(k) for k in ("id", "status", "idea")} for r in open_],
        "requests_note": (
            "This run names no request, and some are open. If it tries one of them, attach it: "
            f"proxy_request(op='update', request_id='R…', run_id='{run_id}', status='trying'), "
            "and tag the next runs' axis [R….k] so the code map shows them on the request."),
    }


def list_() -> dict:
    return ok({"requests": [{k: r.get(k) for k in ("id", "mode", "kind", "status", "idea", "targets")}
                            for r in requests.load()]})


def get(request_id: str) -> dict:
    row = requests.get(request_id)
    if row is None:
        return err(f"No request {requests.normalise(request_id)}.",
                   hint="proxy_request(op='list') names them.")
    return ok({"request": row})
