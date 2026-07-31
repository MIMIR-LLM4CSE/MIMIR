"""Example MIMIR policy plugin — a locked outbound-secret guard.

Copy this into your plugins directory to activate it — `$MIMIR_PLUGINS_DIR` or
`<workspace>/.mimir/plugins/` — and drop the `.example` suffix. It is NOT auto-loaded
from `mimir/examples/`.

A *policy* BLOCKS a tool call before it runs. This one keys off a **capability**
(`EXTERNAL_FETCH`) and generic argument inspection — never a literal tool name — so it
stays robust across tool renames and applies to any server that declares that capability.
Policies are **locked**: the user cannot toggle them off. A check can only ADD a
constraint; returning `None` (abstain) never relaxes a core gate.

See PLUGINS_DETAILED.md for the full authoring guide.
"""
from __future__ import annotations

import json
import re
from typing import Any

from mimir.client.extensions import PolicyCheck, register_policy_check
from mimir.client.context.capabilities import EXTERNAL_FETCH, has_cap

# Heuristic: an argument name or value that looks like a credential being sent outbound.
_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization|bearer)\b")


def _looks_secret(arguments: dict) -> bool:
    for key, value in (arguments or {}).items():
        if _SECRET_RE.search(str(key)):
            return True
        if isinstance(value, str) and _SECRET_RE.search(value):
            return True
    return False


def _no_secret_exfil(agent: Any, tool_name: str, arguments: dict, execution_context: dict | None) -> str | None:
    """Block an outbound-fetch call that appears to carry a secret; return None to abstain."""
    if not has_cap(tool_name, EXTERNAL_FETCH, getattr(agent, "tool_caps", {})):
        return None
    if not _looks_secret(arguments):
        return None
    return json.dumps({
        "status": "error",
        "error": "Blocked by policy: an outbound request appears to carry a credential. "
                 "Remove the secret, or use an approved secrets/proxy capability instead.",
    })


def register() -> None:
    """Register this plugin's descriptors (called for its import side effect)."""
    register_policy_check(PolicyCheck(
        name="no_secret_exfil",
        check=_no_secret_exfil,
        stage="pre_mutation",   # veto early, before the reach/approval gates
        order=10,
    ))


register()
