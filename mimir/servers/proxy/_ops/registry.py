"""Proxy registration ops: list/inspect (read-only) and register/update/unregister."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone

from _ops import _check_name, _with_next, err, ok
from _lib.command import _PARAM_EXT
from _lib.procs import _run_state
from _lib import store
from _lib.store import (
    _load_registry_or_err, _save_registry, _registry_lock,
    _proxy_runs_dir,
    _read_json,
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

    lines.append("## Identity")
    for key in ("version", "arch", "backend", "parallelism", "tags"):
        val = entry.get(key)
        if val:
            lines.append(f"- **{key}:** {val!r}" if isinstance(val, list) else f"- **{key}:** {val}")
    lines.append(f"- **registered_at:** {entry.get('registered_at', '?')}")
    if entry.get("updated_at"):
        lines.append(f"- **updated_at:** {entry['updated_at']}")
    lines.append("")

    lines.append("## Executable")
    lines.append(f"- **executable_path:** `{entry.get('executable_path', '?')}`")
    lines.append(f"- **output_format:** {entry.get('output_format', 'npz')}")
    if entry.get("source_url"):
        lines.append(f"- **source_url:** {entry['source_url']}")
    if entry.get("build_cmd"):
        lines.append(f"- **build_cmd:** `{entry['build_cmd']}`")
    lines.append("")

    if entry.get("input_description"):
        lines.append("## Input")
        lines.append(entry["input_description"])
        lines.append("")
    if entry.get("output_description"):
        lines.append("## Output")
        lines.append(entry["output_description"])
        lines.append("")

    lines.append("## Run Command Template")
    lines.append(f"```\n{entry.get('run_cmd_template', '(none)')}\n```")
    lines.append("")

    pft = entry.get("param_file_template", "") or ""
    if pft.strip():
        fmt = entry.get("param_file_format", "text")
        pfp = entry.get("param_file_path", "") or "(per-run)"
        lines.append(f"## Parameter File ({fmt}, path: {pfp})")
        excerpt = pft[:400] + ("..." if len(pft) > 400 else "")
        lines.append(f"```\n{excerpt}\n```")
        lines.append("")

    # Roofline ceilings
    peak_gf = entry.get("peak_gflops_per_s", 0)
    peak_bw = entry.get("peak_bandwidth_gbytes_per_s", 0)
    if peak_gf or peak_bw:
        lines.append("## Performance Ceilings")
        if peak_gf:
            lines.append(f"- **peak_gflops_per_s:** {peak_gf}")
        if peak_bw:
            lines.append(f"- **peak_bandwidth_gbytes_per_s:** {peak_bw}")
        lines.append("")

    if entry.get("notes"):
        lines.append("## Notes")
        lines.append(entry["notes"])
        lines.append("")

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
    hint = (f"proxy_get(op='proxy', name='{next(iter(reg))}') to inspect one."
            if reg else "proxy_manage(op='register', ...) to register a proxy.")
    return ok(_with_next({"proxies": list(reg.values()), "count": len(reg)}, hint))


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
            m = _read_json(os.path.join(run_dir, "metrics.json"))
            if isinstance(m, dict):
                row["time_s"] = m.get("time_s")
                row["misfit"] = m.get("misfit")
                row["l2_rel"] = m.get("comparison_to_reference", {}).get("l2_rel")
            recent_runs.append(row)

    return ok(_with_next(
        {"proxy": entry, "readme": readme_text, "recent_runs": recent_runs},
        f"proxy_exec(op='run', proxy_name='{name}', confirm=True) to run it."))


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
    if bad := _check_name("name", name):
        return bad
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

    # The one op that may bring the store into existence: registering a proxy is what
    # creates <workspace>/proxy_bench/. Reads (proxy_get) and the other mutations take
    # the lock without create, so none of them leaves a directory behind on a project
    # that has no proxy registered.
    with _registry_lock(create=True):
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
    return ok(_with_next(
        {"registered": entry},
        f"proxy_exec(op='reference', proxy_name='{name}', reference_name='"
        f"{name}_ref', confirm=True) to seal a reference for comparisons."))


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
    return ok(_with_next({"updated": entry},
                         f"proxy_get(op='proxy', name='{name}') to review the entry."))


def unregister(name: str) -> dict:
    with _registry_lock():
        reg, _reg_err = _load_registry_or_err()
        if _reg_err:
            return err(_reg_err)
        if name not in reg:
            return err(f"Proxy '{name}' not found in registry.")
        removed = reg.pop(name)
        _save_registry(reg)
    return ok(_with_next(
        {"unregistered": removed,
         "note": "Run history, optimisation state and snapshots for this proxy are "
                 "PRESERVED. proxy_manage(op='clean', name=..., confirm=True) removes "
                 "them."},
        "proxy_get(op='proxies') to see what remains registered."))


def clean(name: str) -> dict:
    """Delete a proxy's runs, optimisation state and snapshots. Returns what survived.

    ``unregister`` drops the registry entry and keeps everything else, which is correct
    but was the whole story: there was no way to remove a proxy's state at all. Deleting
    the workspace did not do it either, because the store used to live outside it — a
    user who deleted a project and started again was silently resumed into the old
    optimisation, since ``active_session`` still named the proxy and the registry still
    held a run command pointing at a file that no longer existed.

    It deliberately does NOT cascade into references and suites: a sealed reference costs
    real compute and can be shared by a suite this proxy has nothing to do with. Instead
    the response NAMES what it left behind and how to remove it — the property whose
    absence made "I deleted everything and it still remembers" possible to live through
    without ever seeing why.
    """
    removed, kept_refs, kept_suites = [], [], []
    for path, label in (
        (os.path.join(store.runs_dir(), name), "runs"),
        (os.path.join(store.opt_runs_dir(), name), "optimisation state"),
    ):
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                removed.append(label)
            except OSError as exc:
                return err(f"Could not remove {label}: {exc}")

    # The shadow snapshot repository and its fallback copies live beside the store root.
    for path, label in ((os.path.join(store.cache_dir(), "opt.git"), "tree snapshots"),
                        (os.path.join(store.cache_dir(), "tree_snapshots"), "tree snapshots")):
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                if label not in removed:
                    removed.append(label)
            except OSError:
                pass

    if store._resolve_proxy_name("") == name:
        store._clear_active_session()
        removed.append("active-session pointer")

    for d, sink in ((store.refs_dir(), kept_refs), (store.suites_dir(), kept_suites)):
        try:
            sink.extend(sorted(os.listdir(d)))
        except OSError:
            pass

    still_registered = False
    reg, _e = store._load_registry_or_err()
    if not _e:
        still_registered = name in (reg or {})

    kept: list[str] = []
    if still_registered:
        kept.append(f"the registry entry — proxy_manage(op='unregister', name='{name}', "
                    "confirm=True)")
    if kept_refs:
        kept.append("sealed references " + ", ".join(kept_refs)
                    + " — shared with suites; remove by hand from " + store.refs_dir())
    if kept_suites:
        kept.append("benchmark suites " + ", ".join(kept_suites)
                    + " — proxy_manage(op='suite_delete', name=..., confirm=True)")

    return ok(_with_next({
        "cleaned":  name,
        "removed":  removed or ["nothing — no state was found"],
        "kept":     kept or ["nothing"],
        "note": ("Everything else about this proxy is gone. " if not kept else
                 "What is listed under 'kept' still exists and will be found again by a "
                 "new session. "),
    }, "proxy_get(op='proxies') to see what remains registered."))
