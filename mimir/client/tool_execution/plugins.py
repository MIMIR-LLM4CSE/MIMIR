"""Post-tool hooks: user checks run after a successful call, appending to its result.

A hook cannot block, but may write to the blackboard, which the turn-end gates read.
Register at import time with ``register_post_tool(PostToolRule(name, run))``; ``run`` may
be async. Its commands go through the ordinary approval path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..context.execution_context import ExecutionContext
from ..core.extension_set import registration_target
from ..core.ports import ExecutorAgent

# (agent, tool_name, arguments, result, execution_context) -> annotation appended to the
# tool result, or "" / None for nothing. May be a coroutine function.
PostToolFn = Callable[[ExecutorAgent, str, dict[str, Any], str, ExecutionContext | None], Any]


@dataclass(frozen=True)
class PostToolRule:
    name: str
    run: PostToolFn
    order: int = 100     # lower runs first


def register_post_tool(rule: PostToolRule) -> None:
    """Add a post-tool hook to the extension set being loaded (see ``core.extension_set``)."""
    registration_target().add_post_tool(rule)
