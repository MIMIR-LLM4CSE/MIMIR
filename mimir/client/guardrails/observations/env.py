"""What tool calls say about the environment: missing modules, probes, mutations."""
from __future__ import annotations

from __future__ import annotations
import re as _re
from ...context.capabilities import (
    CODE_EXEC,
    ENV_DISCOVERY,
    ENV_MUTATE,
    has_cap,
)
from .call import Call


# Signals that a check/run failed for lack of an importable module rather than a
# code defect. Matches both Python's runtime "ModuleNotFoundError: No module named
# 'x'" and the static import-resolver's "unresolved import: x" phrasing.
_MODULE_NOT_FOUND_RE = _re.compile(
    r"ModuleNotFoundError|No module named|unresolved import|cannot import name|"
    r"could not be resolved",
    _re.IGNORECASE,
)


# Extracts the offending module name (when the message names it) so the
# env-resolution nudge can quote it. Runs only after detection matched.
_MODULE_NAME_RE = _re.compile(
    r"(?:No module named|unresolved import|cannot import name)[:\s]*['\"]?([\w.]+)",
    _re.IGNORECASE,
)


def missing_module(call: Call) -> None:
    """Flag a failed call that reported a missing importable module; a success clears it.

    Feeds the env-resolution nudge: an interpreter mismatch, not a code defect.
    """
    execution_context = call.execution_context
    if call.status == "ok":
        if has_cap(call.tool_name, CODE_EXEC, getattr(call.agent, "tool_caps", None)):
            execution_context.env.unresolved_modules.clear()
        return
    unresolved = execution_context.env.unresolved_modules
    text_fields = (
        call.payload.get("stderr"),
        call.payload.get("stdout"),
        call.payload.get("error"),
        call.payload.get("hint"),
        call.payload.get("message"),
    )
    blob = "\n".join(str(f) for f in text_fields if f)
    if _MODULE_NOT_FOUND_RE.search(blob) is None:
        return
    name_match = _MODULE_NAME_RE.search(blob)
    if name_match:
        unresolved.add(name_match.group(1))
    else:
        # Failure is module-shaped but we could not name the module; record a marker
        # so the nudge still fires.
        unresolved.add("")


def env_probe(call: Call) -> None:
    """Record that the agent enumerated the available runtime environments.

    Capability-driven (no tool name): any tool carrying ENV_DISCOVERY counts. Once set,
    the env-resolution nudge stops firing — the agent has the env list it was steered to.
    """
    execution_context = call.execution_context
    if has_cap(call.tool_name, ENV_DISCOVERY, getattr(call.agent, "tool_caps", None)):
        execution_context.env.env_probed = True


def env_mutation(call: Call) -> None:
    """Record a successful ``ENV_MUTATE`` call as something the cleanup nudge may offer to undo."""
    execution_context = call.execution_context
    if call.status != "ok":
        return
    if not has_cap(call.tool_name, ENV_MUTATE, getattr(call.agent, "tool_caps", None)):
        return
    mutations = execution_context.env.env_mutations
    record = {
        "tool": call.tool_name,
        "installed": call.payload.get("installed") or [],
        "python": call.payload.get("python") or "",
        "path": call.payload.get("path") or "",
        "name": call.payload.get("name") or "",
    }
    if record not in mutations:
        mutations.append(record)
