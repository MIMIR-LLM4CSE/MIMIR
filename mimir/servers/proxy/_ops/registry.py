"""Proxy registration ops: list/inspect (read-only) and register/update/unregister."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from _ops import err, ok
from _lib.command import _PARAM_EXT
from _lib.procs import _run_state
from _lib.store import (
    _load_registry_or_err, _save_registry, _registry_lock,
    _proxy_runs_dir,
)

# Descriptive registration fields carried by the ``metadata`` dict parameter.
# On-disk registry entries keep these flat, so existing registries stay valid.
_METADATA_DEFAULTS: dict = {
    "arch": "",
    "backend": "",
    "parallelism": "",
    "peak_gflops_per_s": 0.0,
    "peak_bandwidth_gbytes_per_s": 0.0,
    "tags": [],
    "version": "",
    "build_cmd": "",
    "source_url": "",
    "notes": "",
    "input_description": "",
    "output_description": "",
    "usage_examples": [],
    # Name of a conserved scalar the proxy emits in its metrics block; when set,
    # runs compared to a reference also report `conservation_residual`.
    "conserved_metric": "",
}


def _check_metadata(metadata: dict | None) -> str | None:
    if not metadata:
        return None
    unknown = sorted(set(metadata) - set(_METADATA_DEFAULTS))
    if unknown:
        return (f"Unknown metadata key(s): {', '.join(unknown)}. "
                f"Valid keys: {', '.join(sorted(_METADATA_DEFAULTS))}.")
    return None


def _proxy_readme(entry: dict) -> str:
    """Generate a structured human-readable documentation string for a proxy entry."""
    lines = []
    name = entry.get("name", "?")
    lines.append(f"# Proxy: {name}")
    lines.append("")

    if entry.get("description"):
        lines.append(f"**Description:** {entry['description']}")
        lines.append("")

    # Core identity
    lines.append("## Identity")
    for key in ("version", "arch", "backend", "parallelism", "tags"):
        val = entry.get(key)
        if val:
            lines.append(f"- **{key}:** {val!r}" if isinstance(val, list) else f"- **{key}:** {val}")
    lines.append(f"- **registered_at:** {entry.get('registered_at', '?')}")
    if entry.get("updated_at"):
        lines.append(f"- **updated_at:** {entry['updated_at']}")
    lines.append("")

    # Executable & build
    lines.append("## Executable")
    lines.append(f"- **executable_path:** `{entry.get('executable_path', '?')}`")
    lines.append(f"- **output_format:** {entry.get('output_format', 'npz')}")
    if entry.get("source_url"):
        lines.append(f"- **source_url:** {entry['source_url']}")
    if entry.get("build_cmd"):
        lines.append(f"- **build_cmd:** `{entry['build_cmd']}`")
    lines.append("")

    # I/O descriptions
    if entry.get("input_description"):
        lines.append("## Input")
        lines.append(entry["input_description"])
        lines.append("")
    if entry.get("output_description"):
        lines.append("## Output")
        lines.append(entry["output_description"])
        lines.append("")

    # Run command
    lines.append("## Run Command Template")
    lines.append(f"```\n{entry.get('run_cmd_template', '(none)')}\n```")
    lines.append("")

    # Param file
    pft = entry.get("param_file_template", "") or ""
    if pft.strip():
        fmt = entry.get("param_file_format", "text")
        pfp = entry.get("param_file_path", "") or "(per-run)"
        lines.append(f"## Parameter File ({fmt}, path: {pfp})")
        excerpt = pft[:400] + ("..." if len(pft) > 400 else "")
        lines.append(f"```\n{excerpt}\n```")
        lines.append("")

    # Roofline
    peak_gf = entry.get("peak_gflops_per_s", 0)
    peak_bw = entry.get("peak_bandwidth_gbytes_per_s", 0)
    if peak_gf or peak_bw:
        lines.append("## Performance Ceilings")
        if peak_gf:
            lines.append(f"- **peak_gflops_per_s:** {peak_gf}")
        if peak_bw:
            lines.append(f"- **peak_bandwidth_gbytes_per_s:** {peak_bw}")
        lines.append("")

    # Notes
    if entry.get("notes"):
        lines.append("## Notes")
        lines.append(entry["notes"])
        lines.append("")

    # Usage examples
    examples = entry.get("usage_examples") or []
    if examples:
        lines.append("## Usage Examples")
        for ex in examples:
            label = ex.get("label", "Example")
            desc  = ex.get("description", "")
            ep    = ex.get("extra_params", "")
            po    = ex.get("param_overrides", {})
            lines.append(f"### {label}")
            if desc:
                lines.append(desc)
            if ep:
                lines.append(f"- extra_params: `{ep}`")
            if po:
                lines.append(f"- param_overrides: `{po}`")
            lines.append("")

    return "\n".join(lines)


# ── read-only ─────────────────────────────────────────────────────────────────

def list_proxies() -> dict:
    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    return ok({"proxies": list(reg.values()), "count": len(reg)})


def inspect_proxy(name: str) -> dict:
    """Full registration details + readme + last-5-runs summary."""
    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    if name not in reg:
        return err(f"Proxy '{name}' not found.",
                   hint="Call proxy_get(op='proxies') to see registered proxies.")
    entry = dict(reg[name])
    entry["available_placeholders"] = [
        "{executable}", "{output_file}", "{param_file}", "{extra_params}",
    ]
    entry["param_overrides_note"] = (
        "Additional placeholders in templates (e.g. {n}, {size}) are filled "
        "from param_overrides at run time."
    )
    readme_text = _proxy_readme(reg[name])

    recent_runs: list[dict] = []
    srd = _proxy_runs_dir(name)
    if os.path.isdir(srd):
        tags = sorted(
            [d for d in os.listdir(srd) if d != "active" and os.path.isdir(os.path.join(srd, d))],
            reverse=True,
        )
        for tag in tags[:5]:
            run_dir = os.path.join(srd, tag)
            rs = _run_state(run_dir)
            row: dict = {"run_id": f"{name}/{tag}", "state": rs["state"],
                         "elapsed_s": rs["elapsed_s"]}
            mp = os.path.join(run_dir, "metrics.json")
            if os.path.isfile(mp):
                try:
                    with open(mp) as fh:
                        m = json.load(fh)
                    row["time_s"] = m.get("time_s")
                    row["misfit"] = m.get("misfit")
                    row["l2_rel"] = m.get("comparison_to_reference", {}).get("l2_rel")
                except (json.JSONDecodeError, OSError):
                    pass
            recent_runs.append(row)

    return ok({"proxy": entry, "readme": readme_text, "recent_runs": recent_runs})


# ── mutations (confirm already checked by the dispatch tool) ──────────────────

def register(
    name: str,
    executable_path: str,
    run_cmd_template: str,
    description: str = "",
    output_format: str = "npz",
    param_file_template: str = "",
    param_file_path: str = "",
    param_file_format: str = "text",
    metadata: dict | None = None,
) -> dict:
    if not name or not re.match(r"^[A-Za-z0-9_\-]+$", name):
        return err("name must be non-empty and contain only [A-Za-z0-9_-].")
    if output_format not in ("npz", "raw_float64", "none"):
        return err(f"Invalid output_format '{output_format}'.",
                   hint="Use: npz, raw_float64, or none.")
    if not run_cmd_template.strip():
        return err("run_cmd_template is required.")
    if param_file_format.lower() not in _PARAM_EXT:
        return err(f"Invalid param_file_format '{param_file_format}'.",
                   hint="Use one of: " + ", ".join(_PARAM_EXT) + ".")
    meta_err = _check_metadata(metadata)
    if meta_err:
        return err(meta_err)

    abs_exe = os.path.abspath(executable_path)
    if not os.path.isfile(abs_exe):
        return err(f"executable_path not found: {abs_exe}",
                   hint="Provide an absolute path to an existing file.")

    with _registry_lock():
        reg, _reg_err = _load_registry_or_err()
        if _reg_err:
            return err(_reg_err)
        entry = {
            "name":                name,
            "executable_path":     abs_exe,
            "run_cmd_template":    run_cmd_template,
            "output_format":       output_format,
            "description":         description,
            "param_file_template": param_file_template,
            "param_file_path":     param_file_path,
            "param_file_format":   param_file_format,
            **{k: (list(v) if isinstance(v, list) else v) for k, v in _METADATA_DEFAULTS.items()},
            **(metadata or {}),
            "registered_at":       datetime.now(timezone.utc).isoformat(),
        }
        reg[name] = entry
        _save_registry(reg)
    return ok({"registered": entry})


def update(
    name: str,
    executable_path: str = "",
    run_cmd_template: str = "",
    description: str = "",
    output_format: str = "",
    param_file_template: str = "",
    param_file_path: str = "",
    param_file_format: str = "",
    metadata: dict | None = None,
) -> dict:
    meta_err = _check_metadata(metadata)
    if meta_err:
        return err(meta_err)
    with _registry_lock():
        reg, _reg_err = _load_registry_or_err()
        if _reg_err:
            return err(_reg_err)
        if name not in reg:
            return err(f"Proxy '{name}' not found.",
                       hint="Call proxy_get(op='proxies') to see registered proxies.")
        entry = reg[name]
        if executable_path:
            abs_exe = os.path.abspath(executable_path)
            if not os.path.isfile(abs_exe):
                return err(f"executable_path not found: {abs_exe}")
            entry["executable_path"] = abs_exe
        if run_cmd_template:
            entry["run_cmd_template"] = run_cmd_template
        if description:
            entry["description"] = description
        if output_format:
            if output_format not in ("npz", "raw_float64", "none"):
                return err(f"Invalid output_format '{output_format}'.")
            entry["output_format"] = output_format
        if param_file_template:
            entry["param_file_template"] = param_file_template
        if param_file_path != "":
            entry["param_file_path"] = param_file_path
        if param_file_format:
            if param_file_format.lower() not in _PARAM_EXT:
                return err(f"Invalid param_file_format '{param_file_format}'.")
            entry["param_file_format"] = param_file_format
        for key, value in (metadata or {}).items():
            entry[key] = value
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        reg[name] = entry
        _save_registry(reg)
    return ok({"updated": entry})


def unregister(name: str) -> dict:
    with _registry_lock():
        reg, _reg_err = _load_registry_or_err()
        if _reg_err:
            return err(_reg_err)
        if name not in reg:
            return err(f"Proxy '{name}' not found in registry.")
        removed = reg.pop(name)
        _save_registry(reg)
    return ok({"unregistered": removed,
               "note": "Run history for this proxy is preserved."})
