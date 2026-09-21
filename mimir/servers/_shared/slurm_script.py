"""Building blocks of the batch scripts MIMIR submits.

Shared because two servers write batch scripts — ``hpc/server_hpc`` for arbitrary
commands and node probes, ``proxy`` for benchmark and optimisation runs — and the
resource request is the part that must not drift between them: a flag one of them
learns (a node constraint, ``--exclusive`` for a clean timing) is a flag both need.
"""

import os
import re
import shlex
import sys


# The repository root: this file is mimir/servers/_shared/slurm_script.py.
MIMIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))

# A Slurm feature expression (`skylake&ib`, `[a100|h100]`, `gpu*2`) and a hostlist
# (`n[01-04],gpu7`). Both land in a `#SBATCH` line, so anything outside these sets —
# whitespace, quotes, `$` — is refused rather than escaped.
_CONSTRAINT_RE = re.compile(r"[A-Za-z0-9_.&|,!*()\[\]:=-]+")
_NODELIST_RE = re.compile(r"[A-Za-z0-9_.,\[\]-]+")


def validate_target(constraint: str = "", nodelist: str = "",
                    nodes: int | None = None, ntasks: int | None = None) -> str | None:
    """An error message for a bad node-targeting argument, or None."""
    if constraint and not _CONSTRAINT_RE.fullmatch(constraint):
        return "Invalid constraint. Use a Slurm feature expression such as 'skylake' or 'a100|h100'."
    if nodelist and not _NODELIST_RE.fullmatch(nodelist):
        return "Invalid nodelist. Use a Slurm hostlist such as 'n012' or 'n[01-04]'."
    for name, value in (("nodes", nodes), ("ntasks", ntasks)):
        if value is not None and (not isinstance(value, int) or value < 1):
            return f"{name} must be a positive integer."
    return None


def sbatch_header(
    *,
    job_name: str,
    partition: str,
    cpus_per_task: int,
    wall_time: str,
    log_file: str,
    mem: str = "",
    gpus: int = 0,
    account: str = "",
    nodes: int | None = None,
    ntasks: int | None = None,
    constraint: str = "",
    nodelist: str = "",
    exclusive: bool = False,
) -> list[str]:
    """Return the ``#!/bin/bash`` + ``#SBATCH`` directive lines (no command body).

    ``nodes``/``ntasks`` left at None are left to the scheduler's default. ``exclusive``
    is what a timing needs: a benchmark sharing its node with someone else's job
    measures the neighbour as much as the code.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]
    if nodes is not None:
        lines.append(f"#SBATCH --nodes={nodes}")
    if ntasks is not None:
        lines.append(f"#SBATCH --ntasks={ntasks}")
    lines += [
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={wall_time}",
    ]
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    lines += [
        f"#SBATCH --output={shlex.quote(log_file)}",
        f"#SBATCH --error={shlex.quote(log_file)}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    if nodelist:
        lines.append(f"#SBATCH --nodelist={nodelist}")
    if exclusive:
        lines.append("#SBATCH --exclusive")
    return lines


def node_python_lines(var: str = "_MIMIR_PY", explicit: str = "") -> list[str]:
    """Shell lines that set ``$var`` to a MIMIR Python that runs on *this* node.

    The server's own interpreter is the wrong default on a compute node: the in-place
    ``.venv`` is built for the login node's OS, and a node on an older distribution
    or another CPU cannot run it. The candidates, in order: the launcher
    ``install.sh`` leaves (``~/.mimir/bin/python``), which starts the venv recorded
    for the machine it runs on; the portable venv named by OS and architecture
    (``.venv-<os>-<arch>``) in this checkout; the server's own interpreter. Each must
    actually start before it is chosen. An *explicit* interpreter is used as given.
    """
    if explicit:
        return [f"{var}={shlex.quote(explicit)}"]
    launcher = '"${MIMIR_STATE_HOME:-$HOME/.mimir}/bin/python"'
    portable = (shlex.quote(os.path.join(MIMIR_ROOT, ".venv-")) +
                "\"$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)\"/bin/python")
    return [
        f'{var}=""',
        f"for _c in {launcher} {portable} {shlex.quote(sys.executable)}; do",
        f'  if [ -x "$_c" ] && "$_c" -c "import sys" >/dev/null 2>&1; then {var}="$_c"; break; fi',
        "done",
    ]
