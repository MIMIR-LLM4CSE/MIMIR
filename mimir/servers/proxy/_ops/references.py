"""Reference-dataset ops: list (read-only) and create (blocking seal)."""

from __future__ import annotations

import os

from . import check_name, with_next, ok
from .._lib.execute import DEFAULT_MAX_OUTPUT_MB, REF_RUN_TIMEOUT, seal_reference
from .._lib.store import refs_dir, entry_or_err, reference_dir, load_ref_metrics, ref_output_path
from ....shared.proxy_store import read_json


def list_references() -> dict:
    if not os.path.isdir(refs_dir()):
        return ok(with_next({"references": [], "count": 0},
                             "proxy_exec(op='reference', ...) to seal a first reference."))

    out = []
    for name in sorted(os.listdir(refs_dir())):
        rd = reference_dir(name)
        if not os.path.isdir(rd):
            continue
        entry: dict = {"name": name}
        cfg = read_json(os.path.join(rd, "config.json"))
        if cfg is not None:
            entry["config"] = cfg
        m = load_ref_metrics(name)
        if m:
            entry["metrics"] = m
        entry["has_field_output"] = ref_output_path(name) is not None
        out.append(entry)

    return ok(with_next(
        {"references": out, "count": len(out)},
        "proxy_exec(op='run', compare_to_reference=..., confirm=True) to score a run."))


def create(
    proxy_name: str,
    reference_name: str,
    extra_params: str = "",
    param_overrides: dict | None = None,
    max_output_mb: int = DEFAULT_MAX_OUTPUT_MB,
    timeout_s: int = REF_RUN_TIMEOUT,
) -> dict:
    """Run a proxy synchronously and seal its output as an immutable reference."""
    if bad := check_name("reference_name", reference_name):
        return bad
    entry, error = entry_or_err(proxy_name)
    if error:
        return error

    result, error = seal_reference(
        entry, reference_name,
        extra_params=extra_params, param_overrides=param_overrides,
        max_output_mb=max_output_mb, timeout_s=timeout_s,
    )
    if error:
        return error

    return ok(with_next(
        result,
        f"proxy_exec(op='run', proxy_name='{proxy_name}', "
        f"compare_to_reference='{reference_name}', confirm=True) to compare "
        "new runs against this reference."))
