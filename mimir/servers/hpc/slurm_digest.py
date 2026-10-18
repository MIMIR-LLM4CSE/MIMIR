"""The stable hardware of every scheduler the workspace reaches, built once per process and host.

Occupancy-free (``slurm_nodes.stable_types``), so it can be kept; ``slurm_schedulers(add=…)``
re-probes. The I/O half of ``slurm_nodes``.
"""

from __future__ import annotations

import shlex

from .slurm_nodes import (
    SCONTROL_NODES_ARGV,
    SINFO_NODES_FORMAT,
    aggregate_node_types,
    parse_pipe_table,
    parse_scontrol_nodes,
    parse_sinfo_nodes,
    stable_signature,
    stable_types,
)
from .._shared.slurm_site import LOCAL, argv_for, name_of, slurm_is_local

import subprocess
from .._shared.clock import utc_stamp

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
            "partitions": [], "signature": "", "source": "", "degraded": ""}


def cluster_digest(host: str = LOCAL, *, timeout: int = _TIMEOUT) -> dict:
    """*host*'s node kinds and partitions, probed; ``source`` names the tool that answered ("" none).

    ``scontrol show node`` first, ``sinfo -N`` as a degraded fallback.
    """
    if host == LOCAL and not slurm_is_local():
        return _empty(host)

    digest = _empty(host)
    stdout = _capture(shlex.join(SCONTROL_NODES_ARGV), host, timeout)
    if stdout is not None:
        nodes = parse_scontrol_nodes(stdout)
        digest["source"] = "scontrol"
    else:
        stdout = _capture(f"sinfo -N -h -o '{SINFO_NODES_FORMAT}'", host, timeout)
        if stdout is None:
            return digest
        nodes = parse_sinfo_nodes(stdout)
        digest["source"] = "sinfo"
        digest["degraded"] = _DEGRADED

    types = aggregate_node_types(nodes) if nodes else []
    # Names and counts are not occupancy; the signature below is computed without them.
    digest["node_types"] = [
        {**stable, "example_nodes": t["example_nodes"], "nodes_total": t["nodes_total"]}
        for stable, t in zip(stable_types(types), types)
    ]
    digest["signature"] = stable_signature(nodes) if nodes else ""

    stdout = _capture(f"sinfo -h -o '{_PARTITIONS_FORMAT}'", host, timeout)
    if stdout is not None:
        rows = parse_pipe_table(stdout, _PARTITIONS_COLUMNS)
        for row in rows:
            row["partition"] = row["partition"].rstrip("*")
        digest["partitions"] = rows
    return digest


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
