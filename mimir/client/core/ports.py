"""The agent as the engine, the guardrails and the tool executor may use it.

:class:`AgentPort` is the whole of what those layers reach for on an agent — nothing
else of ``MimirAgent`` is theirs to call. It is a ``Protocol``, so a test can hand in
any object with these members, and the real agent is checked against it by type, not
by inheritance.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Callable, Protocol

from ..config.toggles import Toggles
from ..context.capabilities import ToolCaps
from ..context.carry import CarryContext
from ..context.execution_context import ExecutionContext
from .extension_set import ExtensionSet
from .turn_state import TurnState

if TYPE_CHECKING:
    from ..guardrails.policy.approval import ApprovalManager


class AgentPort(Protocol):
    # ── What the agent is ──────────────────────────────────────────────────────
    model: str
    mode: str
    session_id: str | None
    workspace_root: str
    # The live tool registry: name -> declared capabilities, and name -> its server.
    tool_caps: dict[str, ToolCaps]
    tool_owner: dict[str, str]
    tools: list[dict[str, Any]]
    sessions: dict[str, Any]
    skills: dict[str, dict[str, Any]]
    resources: dict[str, Any]
    # The MCP sessions' lifetimes, closed with the agent.
    exit_stack: AsyncExitStack
    approvals: ApprovalManager
    extensions: ExtensionSet

    # ── Settings read live, at every step ──────────────────────────────────────
    thinking: bool
    thinking_depth: int
    thinking_budget: int
    temperature: float | None
    context_mode: str
    enforcement: str
    subagent_level: str
    toggles: Toggles
    # Bounds a caller may set (a sub-agent is held below the model's window).
    max_context_tokens: int
    max_answer_tokens: int

    # ── The turn in flight, and the last one that finished ─────────────────────
    turn: TurnState
    # What one query leaves for the next in the session.
    carry: CarryContext

    # ── What a front-end lends the turn (None where it lends nothing) ─────────
    # Messages the user typed while the turn runs, drained at each step boundary.
    poll_steer: Callable[[], list[str]] | None
    # Hand a long run to the front-end, which watches it and wakes the turn after.
    register_background_job: Callable[..., bool] | None
    # The runs the front-end is already watching, so asking about one can be refused.
    watched_background_jobs: Callable[[], list[dict[str, Any]]] | None

    # ── Paths and tool calls ───────────────────────────────────────────────────
    def normalize_workspace_path(self, path: str | None) -> str: ...
    def is_code_filepath(self, path: str) -> bool: ...
    def is_write_tool(self, tool_name: str) -> bool: ...
    def normalize_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def rewrite_tool_for_context(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...
    def get_tool_file_targets(self, tool_name: str, arguments: dict[str, Any]) -> list[str]: ...
    def advertised_tools(self) -> list[dict[str, Any]]: ...
    def skill_enabled(self, name: str) -> bool: ...
    async def run_tool(self, tool_name: str, arguments: dict[str, Any],
                       execution_context: ExecutionContext | None = None,
                       run_auto_validation: bool = True, call_id: str = "",
                       record_observations: bool = True) -> str: ...
    async def auto_validate_written_file(self, path: str,
                                         execution_context: ExecutionContext | None) -> str: ...

    # ── Approvals and the user ─────────────────────────────────────────────────
    def approval_scope(self, tool_name: str, arguments: dict[str, Any]) -> str: ...
    def request_tool_approval(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]: ...
    def request_path_approval(self, paths: list[str], tool_name: str,
                              arguments: dict[str, Any] | None = None) -> tuple[bool, bool]: ...
    def request_user_question(self, questions: list[Any], origin: dict[str, Any] | None = None,
                              timeout_secs: float | None = None) -> dict[str, Any]: ...
    def record_denied_tool_call(self, tool_name: str, arguments: dict[str, Any],
                                execution_context: ExecutionContext | None,
                                note: str = "") -> None: ...
    def denied_tool_result(self, tool_name: str, arguments: dict[str, Any], note: str = "",
                           execution_context: ExecutionContext | None = None) -> str: ...

    # ── The query's state ──────────────────────────────────────────────────────
    def new_execution_context(self, workspace_root: str = "") -> ExecutionContext: ...
    def todo_file(self) -> str: ...
    async def build_system_content(self, active_mode: str) -> str: ...


class HumanIO(Protocol):
    """Whoever answers for the person at the end of a front-end.

    The agent asks through this and nothing else: a tool that needs approval, a path
    outside the workspace, a question for the user. Each front-end puts its own on the
    agent (``agent.human``) — the terminal prompts on stdin, the WebSocket raises a card
    and parks the turn on its answer, a sub-agent relays to the session that delegated
    it, an unattended run approves. It is handed the agent, since what to show about a
    call (its label, its risk, its scope) is the agent's to say.
    """

    def approve_tool(self, agent: AgentPort, tool_name: str, arguments: dict[str, Any],
                     max_attempts: int = 3) -> tuple[bool, str]:
        """Whether *tool_name* may run with *arguments*, and the note to record."""
        ...

    def approve_paths(self, agent: AgentPort, paths: list[str], tool_name: str,
                      arguments: dict[str, Any] | None = None) -> tuple[bool, bool]:
        """Whether a call may reach *paths* outside the workspace: (approved, always)."""
        ...

    def ask(self, agent: AgentPort, questions: list[Any], origin: dict[str, Any] | None = None,
            timeout_secs: float | None = None) -> dict[str, Any]:
        """The user's answers to *questions*: ``{"answers": [...]}``, flagged
        ``timed_out`` when the wait ran out with nothing answered."""
        ...
