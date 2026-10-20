"""The agent as each layer may use it: small ports by need, composed per layer.

A brick says one thing an agent offers (its tool registry, its paths, its approvals…); a
layer is handed the composition it needs (:class:`PolicyAgent`, :class:`ExecutorAgent`…)
and nothing else of ``MimirAgent`` is its to reach. :class:`AgentPort` is every brick,
for the engine that orchestrates them. ``Protocol``s, so a test hands in any object with
these members, and the real agent is checked against each by type.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from ..config.toggles import Toggles
from ..tools.capabilities import ToolCaps
from ..context.carry import CarryContext
from ..context.execution_context import ExecutionContext
from .extension_set import ExtensionSet
from .turn_state import TurnState

if TYPE_CHECKING:
    from ..guardrails.policy.approval import ApprovalManager


# ── Bricks ────────────────────────────────────────────────────────────────────

class ToolRegistry(Protocol):
    """The tools the agent has, what each declares, and what a call names."""

    # name -> declared capabilities, and name -> its server
    tool_caps: dict[str, ToolCaps]
    tool_owner: dict[str, str]
    tools: list[dict[str, Any]]
    skills: dict[str, dict[str, Any]]
    extensions: ExtensionSet

    def advertised_tools(self) -> list[dict[str, Any]]: ...
    def advertised_tools_for_mode(self, active_mode: str | None = None) -> list[dict[str, Any]]: ...
    def model_invocable_skills(self) -> list[tuple[str, str]]: ...
    def skill_enabled(self, name: str) -> bool: ...
    def is_write_tool(self, tool_name: str) -> bool: ...
    def normalize_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def rewrite_tool_for_context(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...
    def get_tool_file_targets(self, tool_name: str, arguments: dict[str, Any]) -> list[str]: ...


class WorkspacePaths(Protocol):
    """Where the agent works, and for which conversation."""

    workspace_root: str
    session_id: str | None

    def normalize_workspace_path(self, path: str | None) -> str: ...
    def is_code_filepath(self, path: str) -> bool: ...


class ToolRunner(Protocol):
    """Running a tool, and checking a file a tool wrote."""

    async def run_tool(self, tool_name: str, arguments: dict[str, Any],
                       execution_context: ExecutionContext | None = None,
                       run_auto_validation: bool = True, call_id: str = "",
                       record_observations: bool = True) -> str: ...
    async def auto_validate_written_file(self, path: str,
                                         execution_context: ExecutionContext | None) -> str: ...


class Approvals(Protocol):
    """Asking the user, and recording what was refused."""

    approvals: ApprovalManager
    # No terminal behind the agent (a detached server): nothing may be prompted mid-run.
    non_interactive: bool

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


class AgentSettings(Protocol):
    """The settings read live, at every step."""

    model: str
    mode: str
    thinking: bool
    thinking_depth: int
    thinking_budget: int
    temperature: float | None
    context_mode: str
    enforcement: str
    subagent_level: str
    toggles: Toggles
    backend: str
    # Bounds a caller may set (a sub-agent is held below the model's window).
    max_context_tokens: int
    max_answer_tokens: int

    def apply_setting(self, name: str, value: Any) -> Any: ...


class TurnHost(Protocol):
    """The turn in flight, what one query leaves the next, and what a front-end lends it."""

    turn: TurnState
    carry: CarryContext
    host: AgentHost

    def new_execution_context(self) -> ExecutionContext: ...
    def todo_file(self) -> str: ...
    async def build_system_content(self, active_mode: str) -> str: ...
    def build_system_content_now(self, active_mode: str) -> str: ...
    def compact_messages(self, middle: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class Connections(Protocol):
    """The MCP sessions and their lifetimes, closed with the agent."""

    sessions: dict[str, Any]
    resources: dict[str, Any]
    exit_stack: AsyncExitStack
    # Added to every server's environment (a sub-agent's session, its copy of the tree).
    server_env: dict[str, str]


# ── Per layer ─────────────────────────────────────────────────────────────────

class NudgeAgent(ToolRegistry, AgentSettings, Protocol):
    """What the nudges read."""


class ObserverAgent(ToolRegistry, WorkspacePaths, TurnHost, Protocol):
    """What the observations read."""


class PolicyAgent(ToolRegistry, WorkspacePaths, Approvals, AgentSettings, Protocol):
    """What the policy gates and the approval layer read."""


class ExecutorAgent(ToolRunner, ToolRegistry, WorkspacePaths, Approvals, AgentSettings,
                    TurnHost, Connections, Protocol):
    """What the tool executor and the validation read."""


class ConnectorAgent(Connections, ToolRegistry, WorkspacePaths, Approvals, AgentSettings, Protocol):
    """What connecting the MCP servers fills in and reads."""


class AgentPort(ToolRegistry, WorkspacePaths, ToolRunner, Approvals, AgentSettings,
                TurnHost, Connections, Protocol):
    """Every brick: the agent as the engine, which orchestrates them, uses it."""


class RunWatching(Protocol):
    """A front-end's watcher of detached runs (``query_engine.jobs.RunWatcher``)."""

    def register(self, descriptor: dict[str, Any], owner: str | None = None) -> bool:
        """Hold a long run and wake the turn after; False means wait it out in-turn."""
        ...

    def watched(self) -> list[dict[str, Any]]:
        """The runs held now, so asking whether one is done can be refused."""
        ...


@dataclass(frozen=True)
class AgentHost:
    """What a front-end lends the agent: the person behind it, a watcher for the runs it
    detaches, and the messages the user types while a turn runs.

    The terminal lends a person only; the WebSocket worker and the sub-agent server lend
    all three, each its own (``MimirAgent.attach``).
    """

    human: HumanIO
    watcher: RunWatching | None = None
    # Drained at each step boundary.
    steer: Callable[[], list[str]] | None = None


class HumanIO(Protocol):
    """Whoever answers for the person at the end of a front-end (``agent.human``).

    Approvals, outside paths and questions all go through it: the terminal prompts, the
    WebSocket raises a card, a sub-agent relays to its parent, an unattended run approves.
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
