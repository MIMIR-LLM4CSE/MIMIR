"""MCP Platform Server: what this host is (CPU, memory, GPU, Slurm, modules, toolchains,
Python environments). Facts only, probed on demand.

The probes are ``host_probe``; the module tree's searchable index is ``module_catalogue``,
kept under ``<STATE_DIR>/platform/modules/``.
"""

import asyncio
import json
import os
import platform
import socket
import sys
import threading
import time

from mcp.server.fastmcp import FastMCP

from ...shared.capabilities import tool_caps, CACHEABLE, ENV_DISCOVERY, PANEL_REPORT
from ...shared.clock import utc_stamp
from ...shared.cpu_facts import execution_context, local_os, machine_signature
from ...shared.responses import ok
from . import host_probe, module_catalogue


mcp = FastMCP(
    "PlatformServer",
    debug=False,
    log_level="ERROR",
)


def _build_profile() -> dict:
    workspace_root = os.getcwd()
    cpu, gpu = host_probe.collect_cpu(), host_probe.collect_gpu()
    return {
        "timestamp": utc_stamp(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "execution_context": execution_context(has_slurm=host_probe.cmd_exists("sinfo")),
        "machine_signature": machine_signature(
            cpu.get("arch", ""), cpu, [d.get("name", "") for d in gpu.get("devices", [])]),
        "os": local_os(),
        "cpu": cpu,
        "march": host_probe.collect_march(),
        "digest": host_probe.build_digest(),
        "memory": host_probe.collect_memory(),
        "gpu": gpu,
        "slurm": host_probe.collect_slurm(),
        "modules": module_catalogue.collect_modules(),
        "toolchains": host_probe.collect_toolchains(),
        "conda_envs": host_probe.collect_conda_envs(),
        "virtualenvs": host_probe.collect_virtualenvs(workspace_root),
    }


@mcp.tool(**tool_caps(kind="modules", caps=[ENV_DISCOVERY]))
def platform_probe() -> dict:
    """Collect and return a fresh profile of THIS host — the machine MIMIR runs on.

    Every fact is probed on demand, so none of it can be stale. It reads the module
    catalogue's summary if one already exists but never builds it — use
    platform_search to search the modules themselves.

    `execution_context` says what this host is to the scheduler: `none` (no Slurm —
    this host is the only machine), `login` (a login node: the compute nodes are
    other machines, described by the HPC node-profile tool, not by this one), or
    `in_allocation` (this host is an allocated compute node, so this profile is that
    node's). `march` is the name `-march=native` resolves to here; pass that name, not
    `native`, when building on one machine for another.
    """
    started = time.perf_counter()
    profile = _build_profile()
    elapsed = time.perf_counter() - started
    return ok({"profile": profile, "elapsed_s": round(elapsed, 3)})


@mcp.tool(**tool_caps(kind="modules", caps=[ENV_DISCOVERY]))
def platform_get_profile() -> dict:
    """Return a fresh profile of THIS host, with live sinfo data.

    On a login node this describes the login node, not the compute nodes jobs run on;
    see `execution_context` in the result.

    Built on demand for the current host, so it is always correct rather than
    remembered. Like platform_probe, it reports the module catalogue's summary
    without building it.
    """
    return ok({"profile": _build_profile(), "sinfo": host_probe.collect_sinfo()})


@mcp.tool(**tool_caps(kind="modules",
    caps=[ENV_DISCOVERY, CACHEABLE],
    label="Searching modules: {query}",
))
def platform_search(query: str, limit: int = 10, refresh: bool = False) -> dict:
    """Search this host's environment-module catalogue by name or by what a module does.

    A site's module tree is hundreds to thousands of entries — far more than fits in a
    context window — so it is indexed once and searched here instead of listed. Ask by
    name ("cuda", "openmpi") or by capability ("parallel hdf5", "fortran compiler with
    openmp"). Every hit carries `load`: the exact string to pass to `module load`.

    The index is built on first use. Only the name-level pass is built while you wait
    (a second or two); descriptions are filled in behind you, so a result carrying
    `catalogue.enriching: true` means an empty description is *not yet known* rather
    than absent. It is then reused until the module tree itself changes.

    Hosts with no module system return an empty result, not an error.

    Args:
        query: A module name, or a description of what you need it to do.
        limit: Maximum hits to return (1-50).
        refresh: Force a rebuild of the index. Only when the user explicitly asks —
            the index invalidates itself when the module tree changes.
    """
    if not isinstance(query, str) or not query.strip():
        return ok({"query": query, "modules": [], "count": 0,
                   "note": "Give a module name or a description of what you need."})
    limit = 10 if not isinstance(limit, int) or limit <= 0 else min(limit, 50)

    catalogue = module_catalogue.ensure_catalogue(refresh=bool(refresh))
    flavour = catalogue.get("flavour", "none")
    if flavour == "none":
        return ok({
            "query": query, "module_system": "none", "modules": [], "count": 0,
            "note": "No Lmod or Tcl Environment Modules on this host, so there is no "
                    "module catalogue to search. Software here is whatever is on PATH "
                    "or in a Python environment.",
        })

    modules = module_catalogue.rank_modules(catalogue.get("modules", []), query, limit)
    payload = {
        "query": query,
        "module_system": flavour,
        # Named "modules", not "results": a payload carrying "results" is read as the
        # end of workspace discovery by the client's workflow observer, and finding a
        # module says nothing about having explored the repository.
        "modules": modules,
        "count": len(modules),
        "total_indexed": catalogue.get("count", 0),
        "catalogue": {
            "tier": catalogue.get("tier", ""),
            "partial": catalogue.get("partial", False),
            "built_at": catalogue.get("built_at", ""),
            "enriching": not catalogue.get("enrich_done", False),
        },
    }
    if modules:
        # note, not hint: `hint` is reserved and responses.ok() strips it from a
        # success payload, so this line never reached the model.
        payload["note"] = "Load one with: module load <load>"
    return ok(payload)


@mcp.tool(**tool_caps(kind="modules", caps=[ENV_DISCOVERY, CACHEABLE]))
def platform_catalogue_status() -> dict:
    """Report the module catalogue's state without building or refreshing it.

    Says which host it describes, which module system, how many entries, how detailed
    they are, and whether enrichment is still running.
    """
    flavour = module_catalogue.module_flavour()
    catalogue = module_catalogue.load_catalogue()
    if catalogue is None:
        return ok({
            "module_system": flavour, "indexed": False,
            "path": module_catalogue.CATALOGUE_FILE,
            "digest": host_probe.build_digest(),
            "note": ("No module system on this host." if flavour == "none"
                     else "Not indexed yet; the first platform_search builds it."),
        })
    fresh = flavour != "none" and catalogue.get("signal") == module_catalogue.catalogue_signal(flavour)
    return ok({
        "module_system": flavour,
        "indexed": True,
        "path": module_catalogue.CATALOGUE_FILE,
        "hostname": catalogue.get("hostname", ""),
        "count": catalogue.get("count", 0),
        "tier": catalogue.get("tier", ""),
        "partial": catalogue.get("partial", False),
        "enriching": module_catalogue.enriching(),
        "built_at": catalogue.get("built_at", ""),
        "signal_fresh": fresh,
        "digest": host_probe.build_digest(),
    })


# A machine's facts do not change during a session, and the panel asks for them every
# time it refreshes. Probed at most this often, per server process.
_PANEL_PROFILE_TTL_SECS = 300
_PANEL_PROFILE: dict = {"at": 0.0, "profile": {}}
_PANEL_LOCK = threading.Lock()


def _build_panel_profile() -> dict:
    """The fields the panel shows. The full profile costs ~15 s (modules, digest, march)."""
    return {
        "timestamp": utc_stamp(),
        "hostname": socket.gethostname(),
        "cpu": host_probe.collect_cpu(),
        "gpu": host_probe.collect_gpu(),
        "memory": host_probe.collect_memory(),
        "slurm": host_probe.collect_slurm(),
    }


def _panel_profile(force: bool = False) -> dict:
    now = time.time()
    if (force or not _PANEL_PROFILE["profile"]
            or now - _PANEL_PROFILE["at"] > _PANEL_PROFILE_TTL_SECS):
        try:
            _PANEL_PROFILE["profile"] = _build_panel_profile()
            _PANEL_PROFILE["at"] = now
        except Exception:
            return dict(_PANEL_PROFILE["profile"])
    return dict(_PANEL_PROFILE["profile"])


@mcp.tool(**tool_caps(kind="modules",
    caps=[PANEL_REPORT, ENV_DISCOVERY],
    read_only=True,
    panel={"section": "Machine", "order": 10, "refresh_arg": "refresh"},
    label="Machine profile",
))
async def platform_panel_report(refresh: bool = False) -> dict:
    """The host MIMIR runs on, in a form a panel can show: a few lines, no probing.

    Deliberately narrow. On a cluster this is the login node — the one machine nothing
    is measured on — so it is a landmark, not the answer to "what will this run on";
    the Cluster section answers that. It sits first all the same: it is the one section
    that is always there and always cheap, so the drawer has something true in it before
    anything has been asked of a scheduler. On a laptop it is the whole story.

    The profile is probed at most once every few minutes (see ``_panel_profile``): the
    panel refreshes on a timer while it is open and whenever a turn ends, and hardware
    does not change between two of those.

    Args:
        refresh: Probe this machine again rather than serving the held profile — a GPU
            that was busy, a module system loaded since. The reader's own refresh.
    """
    # Off the server's loop: a sync tool holds every other call to this server for as
    # long as the probe runs.
    return await asyncio.to_thread(_platform_panel_report, refresh)


def _platform_panel_report(refresh: bool) -> dict:
    with _PANEL_LOCK:
        profile = _panel_profile(force=refresh)
    if not profile:
        return ok({"title": "Machine", "lines": [],
                   "detail": "The machine could not be probed."})

    cpu = profile.get("cpu") or {}
    gpu = profile.get("gpu") or {}
    mem = profile.get("memory") or {}
    slurm = profile.get("slurm") or {}
    devices = gpu.get("devices") or []
    lines = [
        {"label": "host", "value": str(profile.get("hostname", "?"))},
        # The model name is empty on hosts whose lscpu does not print one, so the
        # architecture stands in rather than a blank before the count.
        {"label": "cpu", "value": f"{cpu.get('model') or cpu.get('arch') or '?'}"
                                  f" × {cpu.get('logical_cpus', '?')}"},
    ]
    if mem.get("total_gb"):
        lines.append({"label": "memory", "value": f"{mem['total_gb']} GB"})
    if devices:
        first = devices[0].get("name", "?")
        lines.append({"label": "gpu",
                      "value": f"{len(devices)} × {first}" if len(devices) > 1 else first})
    elif gpu.get("available"):
        lines.append({"label": "gpu", "value": ", ".join(gpu.get("vendors") or ["present"])})
    # No Slurm line here any more: the partitions, the queue and the hardware of the
    # nodes work actually runs on are a section of their own, filled by the server that
    # can read them (hpc_panel_report). Repeating the partition names here said nothing
    # about them and invited the login node's CPU to be read as the cluster's.
    detail = "The host MIMIR runs on."
    if slurm.get("available"):
        detail += " Submitted work runs on the cluster's nodes, not here — see Cluster."
    return ok({"title": "Machine", "lines": lines,
               "detail": f"{detail} Probed {profile.get('timestamp', '?')}."})


if __name__ == "__main__":
    # --profile-json: print this machine's profile and exit, without serving MCP. A
    # Slurm probe job runs this on a compute node, so the node is described by the
    # very collectors that describe the login node — never by a second, thinner probe.
    if "--profile-json" in sys.argv[1:]:
        json.dump(_build_profile(), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        mcp.run()
