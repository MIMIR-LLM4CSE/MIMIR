"""Reference-dataset ops: list (read-only) and create (blocking seal)."""

from __future__ import annotations

import json
import os

from _ops import err, ok
from _lib.execute import _DEFAULT_MAX_OUTPUT_MB, _REF_RUN_TIMEOUT, _seal_reference
from _lib.store import (
    refs_dir,
    _load_registry_or_err,
    _ref_dir, _load_ref_metrics, _ref_output_path,
)


def list_references() -> dict:
    if not os.path.isdir(refs_dir()):
        return ok({"references": [], "count": 0})

    out = []
    for name in sorted(os.listdir(refs_dir())):
        rd = _ref_dir(name)
        if not os.path.isdir(rd):
            continue
        entry: dict = {"name": name}
        cfg_p = os.path.join(rd, "config.json")
        if os.path.isfile(cfg_p):
            try:
                with open(cfg_p) as fh:
                    entry["config"] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        m = _load_ref_metrics(name)
        if m:
            entry["metrics"] = m
        entry["has_field_output"] = _ref_output_path(name) is not None
        out.append(entry)

    return ok({"references": out, "count": len(out)})


def create(
    proxy_name: str,
    reference_name: str,
    extra_params: str = "",
    param_overrides: dict | None = None,
    max_output_mb: int = _DEFAULT_MAX_OUTPUT_MB,
    timeout_s: int = _REF_RUN_TIMEOUT,
) -> dict:
    """Run a proxy synchronously and seal its output as an immutable reference."""
    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    if proxy_name not in reg:
        return err(f"Proxy '{proxy_name}' not registered.",
                   hint="Call proxy_manage(op='register', ...) first.")
    entry = reg[proxy_name]

    result, error = _seal_reference(
        entry, reference_name,
        extra_params=extra_params, param_overrides=param_overrides,
        max_output_mb=max_output_mb, timeout_s=timeout_s,
    )
    if error:
        return error

    result["next_step"] = (
        f"proxy_exec(op='run', proxy_name='{proxy_name}', "
        f"compare_to_reference='{reference_name}', confirm=True) to compare "
        "new runs against this reference."
    )
    return ok(result)
