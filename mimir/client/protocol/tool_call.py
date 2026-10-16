"""What the model sends when it calls a tool: its arguments, and the one it adds.

The wire format of a call, shared by every layer that reads one — the backends that
parse it, the dispatcher that runs it, the server manager that builds the schema the
model writes it against. Depends on nothing above it.
"""
from __future__ import annotations

import json
from typing import Any

# The argument the model's per-call description arrives in. Named here, beside the rest
# of the call's wire format, so the schema builder that adds the parameter and the
# dispatcher that strips it cannot disagree about its spelling.
DOING_ARG = "doing"

# Set on a call whose arguments are not a JSON object — most often a call the model
# was cut off in the middle of, at its output cap. Read as empty arguments, such a call
# ran anyway (a write with no content); the dispatcher answers it with an error instead.
UNPARSED_ARGUMENTS = "__unparsed_arguments__"


def as_dict(obj: Any) -> dict:
    """A provider object (a Pydantic model, an SDK object, a dict) as a plain dict.

    How a message or a tool call is read whichever backend produced it.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def normalize_arguments(arguments: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict, whichever shape the provider sent them in.

    Every backend parses through here, so a malformed call is caught the same way
    whichever one produced it: it comes back as ``{UNPARSED_ARGUMENTS: raw}``.
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {UNPARSED_ARGUMENTS: arguments}
        return parsed if isinstance(parsed, dict) else {UNPARSED_ARGUMENTS: arguments}
    return {}
