"""Where each phase of a Slurm eval run is sent.

One ``sbatch`` carrying both the build and the measurement gives both the same
partition and the same node, which is wrong in both directions: a node dedicated to
GPU simulation has few cores and no business running ``make``, and the node that
compiles fastest is not the hardware anyone wants a number from.

So an eval run has two placements, and this is the one place that decides the
build's — resolved from three sources, the order between them being the whole rule:

    the call's own build_* argument  >  the registration's build_* metadata
                                     >  what the run phase was given

The registration is the term that matters most in practice: the ratchet submits the
same run hundreds of times, and a placement the model must restate every iteration
is one that comes out wrong on some iteration.

``constraint``, ``nodelist`` and ``gpus`` deliberately do NOT fall through from the
run phase. The first two would pin the build to the exact hardware the split exists
to keep it off; the third charges simulation hours to a compiler that is no faster
for holding a GPU. Nor do ``ntasks`` and ``exclusive``: those describe a measurement
— an MPI rank count, and a node to itself so the timing is not measuring a neighbour
— and a compile wants neither.

``resolve_build`` returns None when no build partition is resolved anywhere, and
None is what keeps the old path intact: one job, one ``--phase all``.
"""

from __future__ import annotations

# Fields a build inherits from the run phase when neither the call nor the
# registration names one. Everything else is either build-only or deliberately
# not inherited (see the module docstring).
_INHERITED = ("cpus_per_task", "mem", "wall_time", "account")


def run_placement(
    *,
    partition: str,
    gpus: int = 0,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
    constraint: str = "",
    nodelist: str = "",
    ntasks: int = 1,
    exclusive: bool = False,
) -> dict:
    """The measurement phase's placement, as one dict to pass around and record."""
    return {
        "partition":     partition,
        "gpus":          gpus,
        "cpus_per_task": cpus_per_task,
        "mem":           mem,
        "wall_time":     wall_time,
        "account":       account,
        "constraint":    constraint,
        "nodelist":      nodelist,
        "ntasks":        ntasks,
        "exclusive":     exclusive,
    }


def _first(*candidates):
    """The first candidate that is set (0 and '' both count as unset here)."""
    for value in candidates:
        if value:
            return value
    return None


def resolve_build(
    entry: dict,
    run: dict,
    *,
    build_partition: str = "",
    build_constraint: str = "",
    build_cpus_per_task: int = 0,
    build_mem: str = "",
    build_wall_time: str = "",
    build_gpus: int = 0,
) -> dict | None:
    """Where this proxy's build goes, or None to keep build and run in one job.

    *entry* is the registry entry (its ``build_*`` metadata is the standing default),
    *run* the placement returned by :func:`run_placement`.
    """
    partition = _first(build_partition, entry.get("build_partition", ""))
    if not partition:
        return None

    out = {
        "partition":  partition,
        "constraint": _first(build_constraint,
                             entry.get("build_constraint", "")) or "",
        "nodelist":   "",
        "gpus":       build_gpus or 0,
        "ntasks":     1,
        "exclusive":  False,
    }
    overrides = {
        "cpus_per_task": (build_cpus_per_task,
                          entry.get("build_cpus_per_task", 0)),
        "mem":           (build_mem, entry.get("build_mem", "")),
        "wall_time":     (build_wall_time, entry.get("build_wall_time", "")),
        "account":       ("", ""),
    }
    for field in _INHERITED:
        call_value, registered = overrides[field]
        out[field] = _first(call_value, registered, run.get(field)) or run.get(field)
    return out
