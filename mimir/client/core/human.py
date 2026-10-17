"""The people the agent can ask, for the front-ends that answer for themselves.

``HumanIO`` (``core.ports``) is how the agent reaches a person. The WebSocket front-end
and the sub-agent server bring their own; these are the two that need nothing but the
agent: a terminal, and nobody at all.
"""
from __future__ import annotations

from typing import Any

from .. import human_pause
from ..context.capabilities import label_for
from ..context.tool_display import shorten_display_args


class TerminalHuman:
    """The person at this process's terminal: asked on stdin, answered by ``input()``.

    A stdin that closes (EOF) answers no, so a run with nobody at the terminal fails
    closed rather than proceeding unasked.
    """

    def approve_tool(self, agent: Any, tool_name: str, arguments: dict[str, Any],
                     max_attempts: int = 3) -> tuple[bool, str]:
        verdict: tuple[bool, str] = agent.approvals.request(
            tool_name=tool_name,
            server_name=agent.tool_owner.get(tool_name, "unknown"),
            arguments=arguments,
            max_attempts=max_attempts,
        )
        return verdict

    def approve_paths(self, agent: Any, paths: list[str], tool_name: str,
                      arguments: dict[str, Any] | None = None) -> tuple[bool, bool]:
        """One prompt for the whole call: a command routinely names several outside
        paths, and asking about each is the same decision put to the user three times.

        The header names the action with a short file name; the absolute paths go on
        their own lines, since *where* is the decision being made — the locations being
        authorised are never the part that is shortened.
        """
        short_args = shorten_display_args(tool_name, arguments or {}, agent.tool_caps)
        what = label_for(tool_name, short_args, agent.tool_caps) or tool_name
        listed = "\n".join(f"  {p}" for p in paths)
        try:
            print(f"\n⚠ {what}\n  needs access outside the workspace:\n{listed}",
                  flush=True)
            # Waiting on the user, not on the tool (see human_pause).
            with human_pause.human_pause():
                reply = input("  Allow once [y] / always [a] / deny [N]? ").strip().lower()
        except EOFError:
            return (False, False)
        if reply in ("a", "always"):
            return (True, True)
        if reply in ("y", "yes", "once"):
            return (True, False)
        return (False, False)

    def ask(self, agent: Any, questions: list[Any], origin: dict[str, Any] | None = None,
            timeout_secs: float | None = None) -> dict[str, Any]:
        """No answers: the elicitation that asked maps this to a ``decline``, and the
        model proceeds with its best judgment."""
        return {"answers": []}


class Unattended:
    """Nobody: every tool and every path is approved, no question is answered.

    For a run graded on what it does by itself — a benchmark — where a prompt would
    wait for ever.
    """

    def __init__(self, note: str = "unattended") -> None:
        self.note = note

    def approve_tool(self, agent: Any, tool_name: str, arguments: dict[str, Any],
                     max_attempts: int = 3) -> tuple[bool, str]:
        return True, self.note

    def approve_paths(self, agent: Any, paths: list[str], tool_name: str,
                      arguments: dict[str, Any] | None = None) -> tuple[bool, bool]:
        return True, False

    def ask(self, agent: Any, questions: list[Any], origin: dict[str, Any] | None = None,
            timeout_secs: float | None = None) -> dict[str, Any]:
        return {"answers": []}
