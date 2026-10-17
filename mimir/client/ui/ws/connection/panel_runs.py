"""Putting the runs the watcher holds onto the machines they are running on.

The panel's two halves come from different places and only meet here. What a cluster is
made of is a server's answer, built from a scheduler's own inventory; what is *running*
is this client's state — the descriptors its watcher holds — and no MCP server can see
it. So the join is made on the way out, in the one process that has both.

Done here rather than in the webview so that the renderer stays a renderer: placement is
a rule about data ("a run with no scheduler is on this machine"), and a rule is worth
unit-testing without a browser.

Nothing here names a tool or a server. A group offers itself as a place to put a run by
carrying ``host``, ``local`` or ``nodes``; a run says where it is with ``scheduler`` and
``nodes``, which is whatever its own status op reported. A group that offers none of
those keys simply never claims anything.
"""

from __future__ import annotations

import os

# The workspace registry servers/_shared/slurm_site.py appends to, one host per line.
_HOSTS_FILE = "slurm_hosts"

# A run nothing claimed is not dropped: it goes back to the caller for the flat list the
# panel already shows, because a run the user cannot see is worse than one shown out of
# place.


def _run_nodes(run: dict) -> set[str]:
    """The nodes a run says it is on. ``node`` as well as ``nodes``: a status op that
    reports a single machine is as entitled to its own spelling as one that reports a
    list."""
    out: set[str] = set()
    for key in ("nodes", "node"):
        value = run.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value.strip())
        elif isinstance(value, (list, tuple)):
            out.update(str(v).strip() for v in value if str(v).strip())
    return out


def _claims_host(group: dict, run: dict) -> bool:
    """Whether this host group is where *run* is running.

    A run carrying a scheduler belongs to the host of that name. One carrying none is
    on this machine — a detached shell command has no scheduler to name, and ``local``
    is how a group says it is this machine.
    """
    scheduler = str(run.get("scheduler") or "").strip()
    if scheduler:
        return str(group.get("host") or "") == scheduler
    return bool(group.get("local"))


def _badge(run: dict) -> dict:
    """What a badge shows, and what its disclosure shows when opened.

    A projection rather than the run itself: the panel renders what it is given, and
    giving it the whole descriptor would have it deciding which fields matter.
    """
    origin = run.get("origin") if isinstance(run.get("origin"), dict) else {}
    badge = {
        "job_key": run.get("job_key", "?"),
        "kind": run.get("kind", ""),
        "state": run.get("state", ""),
        "origin": origin,
    }
    for key in ("phase", "percent", "command", "pid", "job_id", "elapsed_s", "scheduler"):
        if run.get(key) not in (None, "", []):
            badge[key] = run[key]
    nodes = sorted(_run_nodes(run))
    if nodes:
        badge["nodes"] = nodes
    return badge


def recorded_hosts_sections(state_dir: str) -> list[dict]:
    """The Cluster section as the workspace registry has it, before any server is up.

    Names only: what each host is made of is a server's probe, and the drawer asks again
    once the agent is ready. Ids match the server's so open rows survive the swap.
    """
    try:
        with open(os.path.join(state_dir, _HOSTS_FILE)) as fh:
            names = list(dict.fromkeys(line.strip() for line in fh if line.strip()))
    except OSError:
        return []
    if not names:
        return []
    return [{
        "section": "Cluster",
        "title": "Cluster",
        "lines": [],
        "groups": [{"id": f"host:{name}", "label": name, "host": name, "local": False,
                    "lines": [{"label": "reached", "value": f"over ssh to {name}"}]}
                   for name in names],
        "detail": "Recorded for this workspace. Hardware appears once this "
                  "conversation's agent is up.",
    }]


def attach_runs(sections: list[dict], runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hang every run on the host and node rows it is running on.

    Returns ``(sections, unplaced)``. A run lands on its host's row, and also on the row
    of the node kind whose nodes include one of its own — so a Slurm job is visible both
    at the cluster and at the machine, which is the two questions a reader has.

    A node kind lists an example of its nodes rather than all of them, so a run on a
    node outside that sample is placed on its host and not on a kind. The host row is
    the one that must never miss a run; which kind of machine it landed on is a detail
    that can be read from the run itself.
    """
    placed: set[str] = set()
    for section in sections:
        for group in section.get("groups") or []:
            if not isinstance(group, dict):
                continue
            host_badges = []
            for run in runs:
                if not _claims_host(group, run):
                    continue
                host_badges.append(_badge(run))
                placed.add(str(run.get("job_key") or ""))
                for kind in group.get("groups") or []:
                    if not isinstance(kind, dict):
                        continue
                    known = {str(n) for n in (kind.get("nodes") or [])}
                    if known & _run_nodes(run):
                        kind.setdefault("badges", []).append(_badge(run))
            if host_badges:
                group["badges"] = host_badges
    unplaced = [run for run in runs
                if str(run.get("job_key") or "") not in placed]
    return sections, unplaced
