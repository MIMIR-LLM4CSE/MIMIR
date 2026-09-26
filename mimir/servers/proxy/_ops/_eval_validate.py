"""Input guards for ``proxy_eval(op='init'|'configure')``.

Pure checks: each answers "why can this session not be set up that way?" with a
sentence the caller can act on, or None. They are split out of the ops so the
ops read as what they persist rather than as a wall of refusals — and because a
refusal has to be readable on its own to be worth writing.
"""

from __future__ import annotations

import os

from _lib.metrics import _VALID_OPT_OPERATORS
from _lib.store import workspace_root

# Computed server-side from a sealed reference, so a requirement on one of these
# can never be satisfied when a case has no reference — refuse it up front.
_REFERENCE_METRICS = ("conservation_residual", "l2_abs", "l2_rel",
                      "linf_abs", "linf_rel")


def _check_optimize_paths(paths: list[str], proxy_source_path: str) -> str | None:
    """The proxy is a HARNESS; the code under optimisation is somewhere else.

    Nothing here is language-specific: an entry is checked for being a file, inside
    the workspace, and not the harness itself. A compiled project lists its sources
    and declares a build_cmd; the harness then exercises the binary the build
    produces. What must not be listed is a generated file, which a build would
    rewrite underneath the snapshot.

    The shape that must be refused rather than discouraged is a harness that
    reproduces the code it is supposed to measure. The ratchet's guarantee (l2_rel
    against a sealed reference) would then hold for the duplicate and for nothing that
    ships — and a standalone script has neither the imports, nor the module-level
    initialisation, nor the memory layout of the package it mirrors, so what gets
    measured is not the thing. Prose does not prevent that; a refusal does.

    Returns an error string, or None when the declared shape is sound.
    """
    root = workspace_root()
    if not paths:
        return ("optimize_paths is required: name the file(s) the ratchet may edit. "
                "The proxy at proxy_source_path is a HARNESS — it runs the code and "
                "prints metrics — and optimize_paths is the code it exercises, which "
                "the harness should use rather than reproduce: import it, link it or "
                "load it, in whatever language it is written.")
    src_real = os.path.realpath(os.path.abspath(proxy_source_path))
    for raw in paths:
        p = os.path.realpath(os.path.abspath(raw))
        if not os.path.isfile(p):
            return f"optimize_paths entry not found: {raw}"
        if os.path.commonpath([p, root]) != root:
            return (f"optimize_paths entry is outside the workspace: {raw}. "
                    "The ratchet only edits code inside the workspace.")
        if p == src_real:
            return ("proxy_source_path cannot be one of optimize_paths. The harness "
                    "must not be its own subject: optimising the program that measures "
                    "means optimising a copy, and the accuracy constraints then say "
                    "nothing about the code you ship. Point optimize_paths at the real "
                    "source(s) and have the harness exercise them.")
    return None


def _check_requirements(requirements: list[dict]) -> str | None:
    for i, req in enumerate(requirements):
        if not req.get("metric"):
            return f"requirements[{i}] is missing 'metric'."
        if req.get("operator") not in _VALID_OPT_OPERATORS:
            return (f"requirements[{i}] has invalid operator '{req.get('operator')}'. "
                    "Use one of: lt, gt, lte, gte, eq.")
        if req.get("threshold") is None:
            return f"requirements[{i}] is missing 'threshold'."
    return None


def _check_reference_requirements(
    requirements: list[dict], suite: dict, entry: dict,
) -> str | None:
    """Reject requirements that can never be satisfied with this setup.

    Reference-dependent metrics (see ``_REFERENCE_METRICS``) are computed
    server-side against a sealed reference; ``conservation_residual``
    additionally needs the registration to name the conserved scalar. Failing
    fast here turns a dead-end session into an actionable setup error.
    """
    needed = sorted({r.get("metric") for r in requirements}
                    & set(_REFERENCE_METRICS))
    if not needed:
        return None
    cases = suite.get("cases") or []
    no_ref = [str(c.get("case_id", "?")) for c in cases
              if not c.get("reference_name")]
    if not cases or no_ref:
        which = ", ".join(no_ref) if no_ref else "(no cases defined)"
        return (f"Requirement(s) {', '.join(needed)} are computed server-side "
                f"against a sealed reference, but benchmark case(s) {which} "
                "have no reference_name — they could never pass. Create the "
                "benchmark with proxy_exec(op='benchmark_create', "
                "reference_params=...), which seals a reference, or add "
                "reference_name to every case. Values printed by the proxy "
                "for these metrics are ignored.")
    if "conservation_residual" in needed and not entry.get("conserved_metric"):
        return ("Requirement conservation_residual needs the proxy "
                "registration to declare which scalar is conserved: "
                "proxy_manage(op='update', metadata="
                "{'conserved_metric': '<metric name>'}, confirm=True).")
    return None
