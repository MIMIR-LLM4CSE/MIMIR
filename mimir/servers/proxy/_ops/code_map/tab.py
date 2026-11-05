"""What the map tab is sent: one session's steps (or several drawn together), the proxies
it can switch between with a line on each, the R&D requests with the runs that tried
them, and, for proxies ticked together, what merging them promises."""
from __future__ import annotations

import os

from ..._lib import history, requests
from ..._lib.layout import merge_file, read_json, resolve_proxy_name
from ..._lib.ratchet import baseline_value, load_best, read_ledger, relative
from ..._lib.store import load_opt_config, opt_ledger_file, opt_sessions
from .. import merge_sources
from .session import delta, merged, session


def proxies() -> list[str]:
    """The sessions the tab can draw: the live ones, then the frozen ones."""
    return opt_sessions() + history.keys()


def answer(proxy_name: str = "", combined: bool = False, selected: list[str] | None = None) -> dict:
    """The steps of *proxy_name* (else the active session's, else the latest session's), of
    the proxies *selected* together, or of every proxy at once when *combined*; with the
    proxies it can switch between, a line on each. Two proxies or more selected also bring
    what merging them promises (``merge_sources.forecast``)."""
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
           "proxies_meta": [_line(n, active) for n in known],
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


def _latest(names: list[str]) -> str:
    """The session that moved last: its ledger, or when it was frozen."""
    def mtime(n: str) -> float:
        try:
            return os.path.getmtime(history.path(n) if history.is_key(n) else opt_ledger_file(n))
        except OSError:
            return 0.0
    return max(names, key=mtime)


def _gain(rel: float | None) -> float | None:
    """1 - best/baseline, whatever the goal."""
    return None if rel is None else round(1 - rel, 6)


def _line(name: str, active: str) -> dict:
    """One line of the picker: where the session ran, its metric, how many runs it settled
    and its gain so far."""
    if history.is_key(name):
        s = history.read(name)
        base, goal = (s.get("baseline") or {}).get("value"), s.get("goal", "min")
        kept = (relative(st["value"], base, goal) for st in s.get("steps", []) if st.get("verdict") == "accept")
        state = s.get("merge") or {}
        return {"name": name, "branch": "", "metric": s.get("metric", ""), "goal": goal,
                "steps": len(s.get("steps", [])), "gain": _gain(min((r for r in kept if r is not None), default=None)),
                "active": False, "frozen": (s.get("frozen") or {}).get("why", "frozen"),
                "merge": bool(state), "sources": [x["proxy"] for x in state.get("sources", [])]}
    cfg = load_opt_config(name) or {}
    rows = read_ledger(name)
    goal = cfg.get("primary_goal", "min")
    state = read_json(merge_file(name)) or {}
    return {"name": name, "branch": cfg.get("branch", ""), "metric": cfg.get("primary_metric", ""), "goal": goal,
            "steps": max(0, len(rows) - bool(cfg.get("baseline_run_id"))),
            "gain": _gain(relative((load_best(name) or {}).get("primary_value"), baseline_value(cfg, rows), goal)),
            "active": name == active, "frozen": "",
            "merge": bool(cfg.get("merge")), "sources": [x["proxy"] for x in state.get("sources", [])],
            # A merge takes the proxies' own sessions, never another merge.
            "mergeable": not cfg.get("merge") and merge_sources.source(name)[0] is not None}


def _steps_of(name: str) -> list[dict]:
    """A proxy's steps, its frozen sessions' first: a request tried before a clean or a
    re-baseline keeps its runs."""
    keys = [k for k in history.keys() if history.proxy_of(k) == name] + [name]
    return [st for k in keys for st in session(k).get("steps", [])]


_REQUEST_FIELDS = ("id", "token", "ts", "mode", "kind", "targets", "idea", "variants",
                   "source", "constraints", "summary", "conclusion", "session")


def requests_view() -> list[dict]:
    """The R&D requests with the runs that tried them: those of their start's proxy whose
    axis names them, each against the value the request started from. A request with runs
    and no conclusion is being tried, whatever its status says."""
    out = []
    for row in requests.load():
        start = row.get("start") or {}
        attached = set(row.get("runs") or [])
        mine = []
        for st in _steps_of(start["proxy"]) if start.get("proxy") else []:
            tag = requests.axis_of(st.get("axis", ""))
            # By its axis tag, or attached to the request after the fact.
            if not (tag and tag[0] == row.get("id")) and st.get("run_id") not in attached:
                continue
            mine.append({"run_id": st["run_id"], "variant": tag[1] if tag else None, "axis": st["axis"],
                         "verdict": st["verdict"], "value": st["value"],
                         "delta": delta(st["value"], start.get("value"))})
        status = row.get("status", "open")
        if mine and not row.get("conclusion") and status in ("open", "studying", "proposed"):
            status = "trying"
        out.append({**{k: row.get(k) for k in _REQUEST_FIELDS},
                    "status": status, "proxy": start.get("proxy", ""), "metric": start.get("metric", ""),
                    "start_value": start.get("value"), "steps": mine})
    return out
