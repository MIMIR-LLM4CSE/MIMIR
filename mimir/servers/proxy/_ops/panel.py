"""The Optimisation panel section: where the current session stands, in this server's
own words — the client renders the lines and knows nothing about proxies.
"""

from __future__ import annotations

import os
import time

from . import eval_observe, eval_session, ok


# A session is pinned to the machine its baseline ran on: a timing describes the
# machine as much as the code, so a run taken elsewhere is settled as "incomparable"
# and moves neither the best nor the stall counter (_ops.eval_session._settle_incomparable).
# That verdict arrives after the run — after the build, the measurement and the wait.
# The same comparison costs nothing before it, which is what this says.
_MACHINE_TTL_SECS = 300
_MACHINE: dict = {"at": 0.0, "value": {}}


def _this_machine() -> dict:
    """This host's identity, probed at most once every few minutes.

    ``local_machine`` shells out to lscpu and nvidia-smi; the panel refreshes on a
    timer, and neither answer changes between two refreshes.
    """
    now = time.time()
    if not _MACHINE["value"] or now - _MACHINE["at"] > _MACHINE_TTL_SECS:
        try:
            from ....shared import cpu_facts
            _MACHINE["value"] = cpu_facts.local_machine()
            _MACHINE["at"] = now
        except Exception:
            return dict(_MACHINE["value"])
    return dict(_MACHINE["value"])


# The ledger summed up for the drawer, keyed on the file's own mtime. The panel refreshes
# on a timer while a turn runs; the ledger changes only when a run settles, which is
# minutes apart at best. Keyed on mtime rather than on a clock so there is no staleness
# to reason about: the moment the file changes, the next refresh re-reads it.
_STEPS: dict = {"key": None, "value": {}}


def _hottest(proxy_name: str, run_id: str) -> dict | None:
    """The costliest workspace file of the best run's profile, and its costliest GPU
    kernel when it ran any, when it was profiled."""
    from .._lib import profiling
    from .._lib.layout import opt_session_runs_dir, read_json
    prof = read_json(profiling.profile_path(os.path.join(opt_session_runs_dir(proxy_name), run_id))) if run_id else None
    files = (prof or {}).get("files") or {}
    kernels = ((prof or {}).get("gpu") or {}).get("kernels") or {}
    parts = []
    if files:
        path, share = next(iter(files.items()))
        parts.append(f"{os.path.basename(path)} · {share:.0%} of the best run")
    if kernels:
        name, row = next(iter(kernels.items()))
        parts.append(f"GPU {profiling.gpu.kernel_base(name) or name} · {row.get('share', 0):.0%}")
    return {"label": "hottest", "value": " · ".join(parts)} if parts else None


# How a ledger verdict is drawn on the curve: kept moves the best; not kept was measured
# and lost; failed measured nothing; elsewhere ran on another machine than the baseline's.
_POINT_KIND = {"accept": "kept", "reject": "not kept", "converged": "not kept",
               "incomparable": "elsewhere"}

# The curve keeps the latest runs: a drawer a few hundred pixels wide cannot tell more apart.
_CURVE_MAX_POINTS = 120


def _curve(rows: list[dict], cfg: dict, base, best) -> dict | None:
    """The session as a curve: every run's value in the order it settled, and the best
    so far after it — the ratchet's incumbent, which only a kept run moves."""
    def num(v):
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    base, best = num(base), num(best)
    incumbent, points = base, []
    for i, row in enumerate(rows, 1):
        value = num(row.get("primary_value"))
        kind = _POINT_KIND.get(row.get("verdict"), "not kept")
        if value is None or row.get("feasible") is False:
            kind = "failed" if kind != "elsewhere" else kind
        if kind == "kept" and value is not None:
            incumbent = value
        points.append({"n": i, "run": row.get("run_id", ""), "axis": row.get("axis") or "",
                       "value": value, "kind": kind, "best": incumbent})
    if not points:
        return None
    kept = sum(1 for p in points if p["kind"] == "kept")
    return {
        "metric": cfg.get("primary_metric", "time_s"),
        "goal": cfg.get("primary_goal", "min"),
        "baseline": base,
        "best": best if best is not None else incumbent,
        "tried": len(points),
        "kept": kept,
        "points": points[-_CURVE_MAX_POINTS:],
    }


def _session_view(proxy_name: str, cfg: dict) -> dict:
    """The session's curve, and the line naming where the best run spends its time; what
    each step changed is the code map's to show, so the section opens it."""
    from .._lib.store import opt_ledger_file
    from .._lib.ratchet import load_best
    try:
        key = (proxy_name, os.path.getmtime(opt_ledger_file(proxy_name)))
    except OSError:
        return {}                       # no ledger yet: nothing tried
    if key == _STEPS["key"]:
        return _STEPS["value"]
    baseline_run = cfg.get("baseline_run_id", "")
    try:
        ledger = eval_observe.read_ledger(proxy_name)
        best_rec = load_best(proxy_name) or {}
    except Exception:
        return {}                       # one section's store costs its lines, never the drawer
    rows = [r for r in ledger if r.get("run_id") != baseline_run]
    base = next((r.get("primary_value") for r in ledger if r.get("run_id") == baseline_run), None)
    view: dict = {}
    chart = _curve(rows, cfg, base, best_rec.get("primary_value"))
    if chart:
        view["chart"] = chart
    hottest = _hottest(proxy_name, best_rec.get("run_id") or baseline_run)
    if hottest:
        view["lines"] = [hottest]
    _STEPS.update({"key": key, "value": view})
    return view


def comparability_lines(cfg: dict, metric: str) -> list[dict]:
    """Where this session's numbers were taken, and whether a run here would join them.

    Only for a metric whose value depends on the machine: an accuracy is reproducible
    anywhere, so pinning would be noise. Nothing is said at all until a baseline has
    pinned a machine — before that there is nothing to differ from.
    """
    from .._lib.ratchet import NOISY_METRICS

    pinned = cfg.get("machine") or {}
    signature = pinned.get("machine_signature") or ""
    if not signature or metric not in NOISY_METRICS:
        return []
    host = pinned.get("host") or pinned.get("cpu_model") or "another machine"
    here = _this_machine()
    if not here.get("machine_signature"):
        return [{"label": "measured on", "value": host}]
    if here["machine_signature"] == signature:
        return [{"label": "measured on", "value": f"{host} (this machine)"}]
    return [
        {"label": "measured on", "value": host},
        {"label": "here", "value": "different machine — a run taken here would be "
                                   "incomparable", "state": "warn"},
    ]


def report() -> dict:
    """The section for the session this workspace is optimising, or a line saying none is."""
    from .._lib.store import load_registry
    from .._lib.layout import resolve_proxy_name
    from .._lib.ratchet import load_best

    name = resolve_proxy_name("")
    if not name:
        return ok({"title": "Optimisation", "lines": [],
                   "detail": "No optimisation session in this workspace."})

    cfg = eval_session.load_opt_config(name) or {}
    state = eval_observe.status(name).get("state", "unknown")
    metric = cfg.get("primary_metric", "time_s")
    view = _session_view(name, cfg)
    lines = [
        {"label": "proxy", "value": name},
        {"label": "state", "value": state,
         "state": "running" if state in ("running", "pending") else ""},
    ]
    if "chart" not in view:
        # Before a step has settled the curve has nothing to draw: the metric and the
        # baseline's value say where it will start.
        best = load_best(name) or {}
        lines.append({"label": "metric", "value": metric})
        if best:
            lines.append({"label": "best so far",
                          "value": f"{best.get('primary_value')} ({best.get('run_id', '?')})"})
    if cfg.get("benchmark_name"):
        lines.append({"label": "benchmark", "value": cfg["benchmark_name"]})
    lines.extend(comparability_lines(cfg, metric))
    lines.extend(view.get("lines", []))
    registry_names = sorted(load_registry().keys())
    if len(registry_names) > 1:
        lines.append({"label": "other proxies",
                      "value": ", ".join(n for n in registry_names if n != name)})
    return ok({"title": "Optimisation", "lines": lines,
               **({"chart": view["chart"], "open": "code_map"} if "chart" in view else {})})
