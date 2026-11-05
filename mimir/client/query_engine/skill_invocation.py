"""The skills a user's query invokes by name: each ``/<skill>`` token, at the start or
after whitespace, folded into the system message before the first step whatever the
model would have judged."""
from __future__ import annotations

import re
from collections.abc import Container

# "/name" opening the text or after whitespace; "a/b" and "/a/b" are paths, not skills.
_INVOCATION_RE = re.compile(r"(?:^|(?<=\s))/([A-Za-z0-9][\w-]*)(?![\w/-])")

_PREAMBLE = ("\n\nSKILL CONTEXT (SUBORDINATE): {name}, invoked by the user. "
             "The base system instructions remain fully authoritative. "
             "Apply this methodology where relevant. ")


def invoked_skills(query: str, skills: Container[str]) -> list[str]:
    """The skills *query* names as ``/<skill>``, in order of first mention."""
    found: list[str] = []
    for match in _INVOCATION_RE.finditer(query or ""):
        name = match.group(1)
        if name in skills and name not in found:
            found.append(name)
    return found


def skill_block(name: str, content: str) -> str:
    """The block one invoked skill adds to the system message."""
    return _PREAMBLE.format(name=name) + content
