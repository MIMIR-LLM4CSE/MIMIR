"""Parameter-file rendering and run/sbatch command building.

Registered proxies are launched from ``run_cmd_template`` with placeholders
``{executable}``, ``{output_file}``, ``{param_file}``, ``{extra_params}``;
any other placeholder is filled from ``param_overrides`` at run time, in both
the command and the parameter-file template.
"""

from __future__ import annotations

import os
import re
import shlex

from _lib import procs

_PARAM_EXT: dict[str, str] = {
    "text":              "txt",
    "json":              "json",
    "yaml":              "yaml",
    "fortran_namelist":  "nml",
    "ini":               "ini",
    "":                  "txt",
}


def _output_ext(output_format: str) -> str:
    return "dat" if output_format == "raw_float64" else "npz"


# ── template expansion ────────────────────────────────────────────────────────

class _KeepMissing(dict):
    """format_map mapping that leaves unknown ``{placeholder}``s untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fill_template(template: str, mapping: dict, param_overrides: dict | None = None) -> str:
    """Expand *template* with *mapping*, then substitute ``param_overrides``.

    Unknown placeholders survive verbatim so overrides (and only overrides)
    can fill them afterwards.
    """
    out = template.format_map(_KeepMissing(mapping))
    if param_overrides:
        for key, value in param_overrides.items():
            out = out.replace("{" + str(key) + "}", str(value))
    return out


def _render_param_file(
    entry: dict,
    run_dir: str,
    output_file: str,
    param_overrides: dict | None,
) -> str | None:
    template = entry.get("param_file_template", "") or ""
    if not template.strip():
        return None

    content = _fill_template(template, {
        "executable":  entry.get("executable_path", ""),
        "output_file": output_file,
        "param_file":  "",
        "extra_params": "",
    })
    if param_overrides:
        for key, value in param_overrides.items():
            placeholder = "{" + str(key) + "}"
            content = content.replace(placeholder, str(value))
            content = re.sub(
                r"(?m)^(\s*" + re.escape(str(key)) + r"\s*=\s*).*$",
                r"\g<1>" + str(value),
                content,
            )
    fixed_path = entry.get("param_file_path", "") or ""
    fmt = entry.get("param_file_format", "") or ""
    ext = _PARAM_EXT.get(fmt.lower(), "txt")
    dest = fixed_path if fixed_path else os.path.join(run_dir, f"params.{ext}")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(content)
    return dest


def _expand_cmd_template(
    entry: dict,
    run_dir: str,
    extra_params: str,
    param_overrides: dict | None = None,
) -> str:
    """Render the param file and expand ``run_cmd_template`` into a command string.

    Single source of truth for the local and sbatch launch paths.
    """
    output_file = os.path.join(run_dir, f"output.{_output_ext(entry.get('output_format', 'npz'))}")
    param_file  = _render_param_file(entry, run_dir, output_file, param_overrides) or ""
    return _fill_template(entry.get("run_cmd_template", ""), {
        "executable":  shlex.quote(entry.get("executable_path", "")),
        "output_file": shlex.quote(output_file),
        "param_file":  shlex.quote(param_file) if param_file else "",
        "extra_params": extra_params or "",
    }, param_overrides)


def _build_run_cmd(
    entry: dict,
    run_dir: str,
    extra_params: str,
    param_overrides: dict | None = None,
) -> list[str]:
    return shlex.split(_expand_cmd_template(entry, run_dir, extra_params, param_overrides))


# ── sbatch script building ────────────────────────────────────────────────────

def _sbatch_header(
    *,
    job_name: str,
    partition: str,
    cpus_per_task: int,
    wall_time: str,
    mem: str,
    log_file: str,
    gpus: int = 0,
    account: str = "",
) -> list[str]:
    """Return the ``#!/bin/bash`` + ``#SBATCH`` directive lines (no command body).

    Shared by ``_build_sbatch`` and the eval Slurm submission path so the
    resource-request preamble (and its quoting) stays consistent.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={wall_time}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --output={shlex.quote(log_file)}",
        f"#SBATCH --error={shlex.quote(log_file)}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    return lines


def _build_sbatch(
    entry: dict,
    run_dir: str,
    extra_params: str,
    *,
    partition: str,
    gpus: int,
    cpus_per_task: int,
    mem: str,
    wall_time: str,
    account: str,
    job_name: str,
    compare_to_reference: str,
    python_exe: str,
    param_overrides: dict | None = None,
) -> str:
    log_file = procs._log_path(run_dir)
    cmd_str  = shlex.join(shlex.split(
        _expand_cmd_template(entry, run_dir, extra_params, param_overrides)))
    lines = _sbatch_header(
        job_name=job_name, partition=partition, cpus_per_task=cpus_per_task,
        wall_time=wall_time, mem=mem, log_file=log_file, gpus=gpus, account=account,
    )
    # Capture wall time + exit code around the solver so the post-run step can
    # apply the same time_s plausibility guard as local runs.
    rc_path   = os.path.join(run_dir, "returncode")
    wall_path = os.path.join(run_dir, "wall_time_ms")
    lines += [
        "",
        "_PROXY_T0=$(date +%s%N)",
        cmd_str,
        "_PROXY_RC=$?",
        "_PROXY_T1=$(date +%s%N)",
        f'echo "$_PROXY_RC" > {shlex.quote(rc_path)}',
        f'echo "$(( (_PROXY_T1 - _PROXY_T0) / 1000000 ))" > {shlex.quote(wall_path)}',
        "",
    ]
    proxy_name = entry.get("name", "")
    ref_arg    = compare_to_reference or ""
    fmt        = entry.get("output_format", "npz")
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    postrun_script = (
        "import sys, os\n"
        f"sys.path.insert(0, {server_dir!r})\n"
        "from _lib.execute import _post_run_finalize\n"
        "_post_run_finalize(\n"
        f"    run_dir={run_dir!r},\n"
        f"    proxy_name={proxy_name!r},\n"
        f"    compare_to_reference={ref_arg!r},\n"
        f"    output_format={fmt!r},\n"
        ")\n"
    )
    postrun_path = os.path.join(run_dir, "postrun.py")
    with open(postrun_path, "w") as _pfh:
        _pfh.write(postrun_script)
    lines += [
        "# --- MCP post-run parsing step ---",
        f"{shlex.quote(python_exe)} {shlex.quote(postrun_path)}",
    ]
    return "\n".join(lines) + "\n"
