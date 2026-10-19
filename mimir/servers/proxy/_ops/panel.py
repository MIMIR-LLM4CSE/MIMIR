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


# The ledger parsed for the drawer, keyed on the file's own mtime. The panel refreshes
# on a timer while a turn runs; the ledger changes only when a run settles, which is
# minutes apart at best. Keyed on mtime rather than on a clock so there is no staleness
# to reason about: the moment the file changes, the next refresh re-reads it.
_AXES: dict = {"key": None, "value": []}


def _axes_lines(proxy_name: str) -> list[dict]:
    """What has been tried on this session, for the drawer.

    Built from the ledger, so the user sees the shape of the search without the agent
    having to report it — and without anyone having to ask for it.
    """
    from .._lib.store import opt_ledger_file
    try:
        key = (proxy_name, os.path.getmtime(opt_ledger_file(proxy_name)))
    except OSError:
        key = None                      # no ledger yet; nothing to cache or to show
    if key is not None and key == _AXES["key"]:
        return _AXES["value"]
    try:
        axes = eval_observe.axes_tried(proxy_name)
    except Exception:
        return []
    if not axes.get("tried"):
        return []
    lines: list[dict] = [{"label": "axes tried",
              "value": f"{axes['tried']}, {len(axes.get('accepted') or [])} accepted"}]
    rejected = [a for a in (axes.get("rejected") or []) if a]
    if rejected:
        lines.append({"label": "ruled out", "value": ", ".join(rejected[-4:])})
    for pair in (axes.get("independent") or [])[-2:]:
        lines.append({"label": "independent", "value": " + ".join(pair)})
    if key is not None:
        _AXES.update({"key": key, "value": lines})
    return lines


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
    from .._lib.store import resolve_proxy_name, load_registry
    from .._lib.ratchet import load_best

    name = resolve_proxy_name("")
    if not name:
        return ok({"title": "Optimisation", "lines": [],
                   "detail": "No optimisation session in this workspace."})

    cfg = eval_session.load_opt_config(name) or {}
    state = eval_observe.status(name).get("state", "unknown")
    best = load_best(name) or {}
    metric = cfg.get("primary_metric", "time_s")
    lines = [
        {"label": "proxy", "value": name},
        {"label": "state", "value": state,
         "state": "running" if state in ("running", "pending") else ""},
        {"label": "metric", "value": metric},
    ]
    if best:
        lines.append({"label": "best so far",
                      "value": f"{best.get('primary_value')} ({best.get('run_id', '?')})"})
    if cfg.get("benchmark_name"):
        lines.append({"label": "benchmark", "value": cfg["benchmark_name"]})
    lines.extend(comparability_lines(cfg, metric))
    lines.extend(_axes_lines(name))
    registry_names = sorted(load_registry().keys())
    if len(registry_names) > 1:
        lines.append({"label": "other proxies",
                      "value": ", ".join(n for n in registry_names if n != name)})
    return ok({"title": "Optimisation", "lines": lines})
