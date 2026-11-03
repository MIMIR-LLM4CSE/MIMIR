"""The stable hardware of every scheduler the workspace reaches, built once per process and host.

Occupancy-free (``slurm_nodes.stable_types``), so it can be kept; ``slurm_schedulers(add=…)``
re-probes. The I/O half of ``slurm_nodes``.
"""

from __future__ import annotations

import shlex

from .slurm_nodes import (
    SCONTROL_NODES_ARGV,
    SCONTROL_PARTITIONS_ARGV,
    SINFO_NODES_FORMAT,
    SQUEUE_LOAD_FORMAT,
    aggregate_node_types,
    parse_pipe_table,
    parse_private_data,
    parse_scontrol_nodes,
    parse_scontrol_partitions,
    parse_sinfo_nodes,
    parse_squeue_load,
    stable_signature,
    stable_types,
    usable_nodes,
    usable_partitions,
)
from ...shared.slurm_site import LOCAL, argv_for, name_of, slurm_is_local

import subprocess
from ...shared.clock import utc_stamp

# Long enough for ``scontrol show node`` on a cluster of a few thousand nodes, short
# enough that a host which has stopped answering does not hold up a listing built from
# several of them.
_TIMEOUT = 20

# The partition columns, in the order the format string asks for them. One fact in two
# halves, kept adjacent for the same reason ``slurm_nodes`` keeps its own pair that way.
_PARTITIONS_FORMAT = "%P|%a|%l|%D|%c|%m"
_PARTITIONS_COLUMNS = [
    "partition", "availability", "time_limit", "nodes",
    "cpus_per_node", "mem_mb_per_node",
]

_DEGRADED = "scontrol unavailable: architecture and node features are unknown."

# One digest per host, for the life of the process. Failures are not entered: a cluster
# that was down for one call must not be silent until the server restarts.
_MEMO: dict[str, dict] = {}



def _capture(script: str, host: str, timeout: int) -> str | None:
    """*script*'s stdout on *host* through a login shell (``slurm_site.argv_for``), or None."""
    try:
        res = subprocess.run(
            argv_for(script, host=host),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 else None


def _empty(host: str) -> dict:
    return {"scheduler": name_of(host), "built_at": utc_stamp(), "node_types": [],
            "partitions": [], "signature": "", "source": "", "degraded": "",
            "left_out": {"nodes": 0, "partitions": {}}, "private_jobs": None}


def _read_nodes(host: str, timeout: int) -> tuple[list[dict] | None, str]:
    """*host*'s nodes and the tool that gave them: scontrol, else sinfo; ``(None, "")`` if neither."""
    stdout = _capture(shlex.join(SCONTROL_NODES_ARGV), host, timeout)
    if stdout is not None:
        return parse_scontrol_nodes(stdout), "scontrol"
    stdout = _capture(f"sinfo -N -h -o '{SINFO_NODES_FORMAT}'", host, timeout)
    if stdout is None:
        return None, ""
    return parse_sinfo_nodes(stdout), "sinfo"


# One login shell for the facts access is decided from. The separator stands between
# sections, so a command that printed nothing leaves its section empty.
_SECTION = "@@mimir@@"
_ACCESS_SCRIPT = (f"{shlex.join(SCONTROL_PARTITIONS_ARGV)} 2>/dev/null; echo {_SECTION}; "
                  "sacctmgr -nP show assoc where user=\"$USER\" format=account "
                  f"2>/dev/null; echo {_SECTION}; id -un; echo {_SECTION}; "
                  "scontrol show config 2>/dev/null | grep -i '^PrivateData'")


def _read_access(host: str, timeout: int) -> dict:
    """What decides access on *host*: partition ``restrictions``, the user's ``accounts``
    (None if not known), ``user``, and ``private_jobs`` (PrivateData hides others' jobs)."""
    stdout = _capture(_ACCESS_SCRIPT, host, timeout) or ""
    sections = stdout.split(_SECTION + "\n")
    if len(sections) != 4:
        return {"restrictions": {}, "accounts": None, "user": "", "private_jobs": None}
    # No association at all is a site without accounting rather than a user with no
    # right anywhere, so it is read as "not known".
    accounts = {a.strip() for a in sections[1].splitlines() if a.strip()} or None
    return {"restrictions": parse_scontrol_partitions(sections[0]), "accounts": accounts,
            "user": sections[2].strip(), "private_jobs": parse_private_data(sections[3])}


def cluster_digest(host: str = LOCAL, *, timeout: int = _TIMEOUT) -> dict:
    """*host*'s node kinds and partitions, probed; ``source`` names the tool that answered ("" none).

    ``scontrol show node`` first, ``sinfo -N`` as a degraded fallback. Only the nodes
    and partitions this user can submit to are kept; ``left_out`` counts the rest.
    ``private_jobs`` is True where squeue shows this user only their own jobs.
    """
    if host == LOCAL and not slurm_is_local():
        return _empty(host)

    digest = _empty(host)
    nodes, digest["source"] = _read_nodes(host, timeout)
    if nodes is None:
        return digest
    if digest["source"] == "sinfo":
        digest["degraded"] = _DEGRADED

    stdout = _capture(f"sinfo -h -o '{_PARTITIONS_FORMAT}'", host, timeout)
    rows = parse_pipe_table(stdout, _PARTITIONS_COLUMNS) if stdout is not None else []
    for row in rows:
        row["partition"] = row["partition"].rstrip("*")
    access = _read_access(host, timeout)
    digest["private_jobs"] = access["private_jobs"]
    usable: set[str] | None = None
    if rows:
        usable_names, left_out = usable_partitions(
            rows, access["restrictions"], access["accounts"], access["user"])
        usable = set(usable_names)
        rows = [row for row in rows if row["partition"] in usable]
        digest["left_out"]["partitions"] = left_out
    digest["partitions"] = rows

    kept = usable_nodes(nodes, usable)
    digest["left_out"]["nodes"] = len(nodes) - len(kept)
    types = aggregate_node_types(kept) if kept else []
    # Names and counts are not occupancy; the signature below is computed without them.
    digest["node_types"] = [
        {**stable, "example_nodes": t["example_nodes"], "nodes_total": t["nodes_total"]}
        for stable, t in zip(stable_types(types), types)
    ]
    digest["signature"] = stable_signature(kept) if kept else ""
    return digest


def usable_set(digest: dict) -> set[str] | None:
    """The partitions *digest* found usable; None when sinfo did not list them."""
    return {p["partition"] for p in digest["partitions"]} if digest["partitions"] else None


def cluster_load(host: str = LOCAL, *, usable: set[str] | None = None,
                 timeout: int = _TIMEOUT) -> dict | None:
    """What is free on *host* now and what its queue wants: never kept. None if no answer.

    ``nodes`` are the usable ones (:func:`usable_nodes` over *usable*); ``queue`` is
    ``parse_squeue_load``'s, None when squeue did not answer.
    """
    nodes, source = _read_nodes(host, timeout)
    if nodes is None:
        return None
    stdout = _capture(f"squeue -h -t PENDING,RUNNING -o '{SQUEUE_LOAD_FORMAT}'",
                      host, timeout)
    return {"source": source, "read_at": utc_stamp(),
            "nodes": usable_nodes(nodes, usable),
            "queue": parse_squeue_load(stdout) if stdout is not None else None}


def cached_cluster_digest(host: str = LOCAL, *, timeout: int = _TIMEOUT) -> dict:
    """:func:`cluster_digest`, kept for the process once *host* has answered."""
    held = _MEMO.get(host)
    if held is not None:
        return held
    digest = cluster_digest(host, timeout=timeout)
    if digest["source"]:
        _MEMO[host] = digest
    return digest


def forget_cluster_digest(host: str | None = None) -> None:
    """Drop the digest held for *host*, or all of them (this process only)."""
    if host is None:
        _MEMO.clear()
    else:
        _MEMO.pop(host, None)
