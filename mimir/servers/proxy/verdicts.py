"""An agent's verdict on a measured run reaches the ratchet: ``report_verdict`` naming a
proxy run settles its tradeoff ('pass' accepts it, 'rejected' rejects it), or takes back
the best ('rejected' on the run that is the best, say for breaking a directive)."""

from __future__ import annotations

import os

from ._lib.layout import opt_runs_dir
from ._lib.ratchet import load_best, read_ledger

_DECISION = {"pass": "accept", "rejected": "reject"}


def settle(run: str, verdict: str, reason: str) -> dict | None:
    """The judgement *verdict* makes of the optimisation run *run*, None when it names none
    or leaves the ratchet as it is."""
    decision = _DECISION.get(verdict)
    if not run or decision is None:
        return None
    try:
        sessions = sorted(os.listdir(opt_runs_dir()))
    except OSError:
        return None
    for name in sessions:
        row = next((r for r in read_ledger(name) if r.get("run_id") == run), None)
        if row is None:
            continue
        is_best = (load_best(name) or {}).get("run_id") == run
        if row.get("verdict") != "tradeoff" and not (decision == "reject" and is_best):
            return None
        from ._ops._eval_ratchet import judge
        return judge(name, run, decision, reason)
    return None
