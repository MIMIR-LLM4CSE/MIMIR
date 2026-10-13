"""The stable half of a scheduler's hardware, for every scheduler the workspace reaches.

A cluster's makeup — the kinds of machine it has, their architectures and GPUs, the
partitions they sit in — is what a submission is sized against, and it is also what
nothing in the toolkit knew about a scheduler on the far side of an ``ssh``. Asked per
call it costs two round trips every time, so it was asked rarely and chosen without;
the usual shape of that mistake is a binary built on the login node and submitted to
nodes of a different architecture, which dies on an illegal instruction after the queue
wait rather than before it.

**Occupancy-free by construction**, which is what makes it keepable. The projection is
``slurm_nodes.stable_types``: what the machines *are*, never what is running on them.
So this can be built once and reused, while ``slurm_nodes(scheduler=…)`` stays a live
read — a cached answer there would turn the sizing tool into a stale one, which is the
opposite of what either exists for.

**Built once per process, per host, and held in memory.** There is no cheap fingerprint
for a cluster on the other side of an ``ssh``: asking whether the hardware changed costs
exactly what rebuilding costs, so the stat-hash trick the module catalogue uses has no
analogue here. A process is cheap to restart and a site adds nodes in months, not
minutes, so the memo is the whole freshness mechanism and there is no TTL to consult.
``slurm_schedulers(add=…)`` is the explicit re-probe. Note the pairing: *which* clusters
exist is a workspace fact that outlives any conversation (``slurm_site``), while *what
they are made of* is relearned per process.

The I/O half of ``slurm_nodes``, which stays pure. This module knows that a scheduler
can be on another host; the parsers it calls do not.
"""

from __future__ import annotations

import shlex
from datetime import datetime, timezone

from slurm_nodes import (
    SCONTROL_NODES_ARGV,
    SINFO_NODES_FORMAT,
    aggregate_node_types,
    parse_pipe_table,
    parse_scontrol_nodes,
    parse_sinfo_nodes,
    stable_signature,
    stable_types,
)
from slurm_site import LOCAL, argv_for, name_of, slurm_is_local

import subprocess

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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _capture(script: str, host: str, timeout: int) -> str | None:
    """*script*'s stdout from *host*, or None if it did not answer.

    Through :func:`slurm_site.argv_for`, so a login shell runs it locally as well as
    remotely. That is not symmetry for its own sake: on a site where Slurm arrives from
    a modulefile, the server process's own ``PATH`` has no ``scontrol``, and a probe
    asking about that ``PATH`` would report no cluster on a machine that has one.

    Every script here is a module constant — nothing a caller passes reaches a shell.
    """
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
    return {"scheduler": name_of(host), "built_at": _now(), "node_types": [],
            "partitions": [], "signature": "", "source": "", "degraded": ""}


def cluster_digest(host: str = LOCAL, *, timeout: int = _TIMEOUT) -> dict:
    """What *host*'s compute nodes and partitions are. Probes; see :func:`cached_cluster_digest`.

    ``source`` says which tool answered, and is ``""`` when none did — an empty
    ``node_types`` therefore never has to be read as "this cluster has no nodes".

    ``scontrol show node`` first, since it is the only one carrying the architecture,
    and ``sinfo -N`` when a site restricts it: a degraded description names its own gaps
    in ``degraded``, where no description at all would have the caller guess.

    The local scheduler is settled by :func:`slurm_site.slurm_is_local` before anything
    runs. A workstation with no Slurm is the common case for that branch, and it must
    not pay two login shells to be told so on every call.
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

    digest["node_types"] = stable_types(aggregate_node_types(nodes)) if nodes else []
    digest["signature"] = stable_signature(nodes) if nodes else ""

    stdout = _capture(f"sinfo -h -o '{_PARTITIONS_FORMAT}'", host, timeout)
    if stdout is not None:
        rows = parse_pipe_table(stdout, _PARTITIONS_COLUMNS)
        for row in rows:
            row["partition"] = row["partition"].rstrip("*")
        digest["partitions"] = rows
    return digest


def cached_cluster_digest(host: str = LOCAL, *, timeout: int = _TIMEOUT) -> dict:
    """:func:`cluster_digest`, built once for *host* and kept for the life of the process.

    Only an answer is kept. A host that said nothing is probed again next time, because
    the alternative is a cluster that was briefly unreachable reading as featureless
    until the server restarts — and a listing that quietly describes nothing is worse
    than one that is slow once.
    """
    held = _MEMO.get(host)
    if held is not None:
        return held
    digest = cluster_digest(host, timeout=timeout)
    if digest["source"]:
        _MEMO[host] = digest
    return digest


def forget_cluster_digest(host: str | None = None) -> None:
    """Drop what is held for *host*, or all of it. The explicit re-probe.

    This reaches one process. A sibling server may still hold a digest for a host that
    has since been dropped from the registry, which costs nothing and needs no
    cross-process invalidation: the memo is only ever read for hosts
    :func:`slurm_site.hosts` returns.
    """
    if host is None:
        _MEMO.clear()
    else:
        _MEMO.pop(host, None)
