from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import json
import logging
import os
import re
import sys
from collections.abc import Callable
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..protocol import human_pause
from ..tools.capabilities import infer_tool_caps
from ..protocol.tool_call import DOING_ARG
from ..config import paths
from ..config.paths import GLOBAL_STATE_DIR, SERVER_BASE
from ...shared.launcher import launch_argv
from ..config.limits import USER_QUESTION_TIMEOUT_SECS
from ...shared.state_paths import scratch_home
from ..core.ports import AgentPort

logger = logging.getLogger(__name__)


# A Google-style ``Args:`` block: the section header, then one entry per parameter as
# ``name: text`` (an optional ``(type)`` between them), continued by more-indented lines.
_ARGS_HEADER_RE = re.compile(r"^[ \t]*Args:[ \t]*$")
# One entry may name SEVERAL parameters that share a description — ``run_a, run_b:``,
# ``input_fmt, output_fmt:`` — which is how these docstrings are actually written.
_ARGS_ENTRY_RE = re.compile(
    r"^([ \t]+)(\*{0,2}\w+(?:[ \t]*,[ \t]*\*{0,2}\w+)*)"
    r"[ \t]*(?:\([^)]*\))?[ \t]*:[ \t]*(.*)$"
)
# Any other Google-style section ends the Args block.
_DOC_SECTION_RE = re.compile(
    r"^[ \t]*(Args|Returns?|Yields?|Raises|Examples?|Notes?|Attributes)[ \t]*:[ \t]*$"
)


def _parse_args_block(doc: str) -> tuple[list[tuple[list[str], str, int, int]], int, int] | None:
    """``(entries, first, end)`` of the ``Args:`` block of *doc*, or None.

    Each entry is ``(names, text, start, stop)``; all bounds are line indices, ends excluded.
    """
    lines = (doc or "").splitlines()
    for i, line in enumerate(lines):
        if _ARGS_HEADER_RE.match(line):
            break
    else:
        return None
    entries: list[list] = []
    indent = ""
    end = len(lines)
    for j in range(i + 1, len(lines)):
        line = lines[j]
        if _DOC_SECTION_RE.match(line):
            end = j
            break
        m = _ARGS_ENTRY_RE.match(line)
        if m:
            indent = m.group(1)
            names = [n.strip().lstrip("*") for n in m.group(2).split(",")]
            entries.append([names, m.group(3).strip(), j, j + 1])
            continue
        if not line.strip():
            continue
        # A more-indented line continues the entry above it (and every parameter that
        # entry named). A line dedented past the entries has left the block. A line at
        # entry indent that is not an entry is skipped rather than ending the block:
        # one unparseable line must not discard every entry after it.
        depth = len(line) - len(line.lstrip())
        if entries and depth > len(indent):
            entries[-1][1] = (entries[-1][1] + " " + line.strip()).strip()
            entries[-1][3] = j + 1
        elif depth < len(indent):
            end = j
            break
    return [tuple(e) for e in entries], i, end


def _args_block_descriptions(doc: str) -> dict[str, str]:
    """Parse a docstring's ``Args:`` block into ``{parameter: description}``.

    Best-effort by design: anything it cannot read yields no entry rather than an
    error. It runs once per tool at server registration, and a malformed docstring
    must never stop a tool being registered.
    """
    parsed = _parse_args_block(doc)
    if not parsed:
        return {}
    return {n: text for names, text, _, _ in parsed[0] for n in names if text}


def _description_without_args_block(doc: str, parameters: dict) -> str:
    """*doc* minus the ``Args:`` entries the schema already carries word for word.

    Otherwise every parameter is sent twice. An entry stays if any parameter it names has
    a different or missing schema text; never raises.
    """
    try:
        parsed = _parse_args_block(doc)
        if not parsed:
            return doc
        entries, first, end = parsed
        props = parameters.get("properties") if isinstance(parameters, dict) else None
        if not entries or not isinstance(props, dict):
            return doc

        def carried(names: list[str], text: str) -> bool:
            return bool(text) and all(
                isinstance(props.get(n), dict) and props[n].get("description") == text
                for n in names
            )

        drop: set[int] = set()
        for names, text, start, stop in entries:
            if carried(names, text):
                drop.update(range(start, stop))
        if not drop:
            return doc
        lines = doc.splitlines()
        if all(carried(names, text) for names, text, _, _ in entries):
            drop.update(range(first, end))  # nothing left under the header
        kept = [line for k, line in enumerate(lines) if k not in drop]
        return "\n".join(kept).rstrip()
    except Exception:  # pragma: no cover - a docstring must never break registration
        logger.debug("_description_without_args_block: a best-effort step failed and was skipped", exc_info=True)
        return doc


# The ``doing`` argument every tool takes: the model's own sentence about this call, shown
# on its row. The dispatcher strips it before execution. Never ``required``, so a
# validating backend does not refuse a call without it.
_DOING_PROPERTY = {
    "type": "string",
    "description": (
        "What this particular call does, in under 15 words, as a phrase: \"searching "
        "for the off-by-one bound\", \"running the row-display tests\", \"submitting "
        "the job on genoa\". Shown to the user beside the tool while the call runs, so "
        "it names the concrete thing this call is for — not the tool's category, and "
        "not a restatement of the arguments."
    ),
}


def _schema_for_model(tool: Any) -> dict:
    """The input schema the model sees: ``doing`` added, ``Args:`` texts lifted into it.

    A constraint stated only in prose is not followed; what the schema says is. An explicit
    ``Field(description=...)`` always wins. Returns a deep copy: ``infer_tool_caps`` reads
    the original.
    """
    schema = getattr(tool, "inputSchema", None)
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    schema = copy.deepcopy(schema)
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = schema["properties"] = {}
    # A server that declares its own `doing` keeps it, like a hand-written description.
    props.setdefault(DOING_ARG, dict(_DOING_PROPERTY))
    try:
        described = _args_block_descriptions(getattr(tool, "description", "") or "")
    except Exception:  # pragma: no cover - a docstring must never break registration
        logger.debug("_schema_for_model: a best-effort step failed and was skipped", exc_info=True)
        return schema
    for name, prop in props.items():
        if isinstance(prop, dict) and not prop.get("description") and described.get(name):
            prop["description"] = described[name]
    return schema


def _bind_question_handler(ask: Any):
    """*ask* as a three-argument callable (questions, asker, deadline), dropping what it does not take."""
    try:
        sig = inspect.signature(ask)
    except (TypeError, ValueError):   # a builtin or C callable: pass the questions only
        return lambda questions, origin, timeout_secs: ask(questions)

    def _accepts(name: str) -> bool:
        try:
            sig.bind(["q"], **{name: None})
        except TypeError:
            return False
        return True

    takes = tuple(n for n in ("origin", "timeout_secs") if _accepts(n))

    def _call(questions: Any, origin: Any, timeout_secs: Any):
        extra = {"origin": origin, "timeout_secs": timeout_secs}
        return ask(questions, **{n: extra[n] for n in takes})

    return _call


def _make_elicitation_callback(agent: AgentPort):
    """Bridge MCP elicitation requests (``ask_user_question``) to ``agent.request_user_question``.

    The blocking handler runs in an executor, with the wait marked as human time on the
    loop thread (``human_pause`` is thread-local). It gets ``USER_QUESTION_TIMEOUT_SECS``;
    plan approval calls the same handler without a deadline.
    """

    async def _callback(
        _context: Any,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        schema = getattr(params, "requestedSchema", None) or {}
        spec = schema.get("x_mimir") if isinstance(schema, dict) else None
        if not isinstance(spec, dict) or spec.get("kind") != "user_question":
            # Not a question we know how to render — decline rather than guess.
            return types.ElicitResult(action="decline")

        questions = spec.get("questions") or []
        if not isinstance(questions, list) or not questions:
            return types.ElicitResult(action="decline")

        declared = spec.get("timeout_secs")
        timeout_secs = (float(declared)
                        if isinstance(declared, (int, float)) and declared > 0
                        else None)

        loop = asyncio.get_running_loop()
        try:
            # With the caller's context: the handler reads which tool call is asking
            # (query_engine.deferral), and a bare executor thread starts from none.
            with human_pause.human_pause():
                result = await loop.run_in_executor(
                    None,
                    contextvars.copy_context().run,
                    _bind_question_handler(agent.request_user_question),
                    questions,
                    # Who is asking, when the asker is not the turn the user is watching:
                    # a sub-agent names itself here so the card can say so. Absent for
                    # the ordinary case, where the asker IS the turn on screen.
                    spec.get("origin") if isinstance(spec.get("origin"), dict) else None,
                    # How long the asker is willing to wait. Declared per question rather
                    # than set here, because the same channel carries approvals and plan
                    # decisions, and neither may ever expire.
                    timeout_secs,
                )
        except Exception:
            logger.debug("_callback: a best-effort step failed and was skipped", exc_info=True)
            return types.ElicitResult(action="cancel")

        result = result or {}
        answers = list(result.get("answers") or [])
        if not answers:
            if not (result or {}).get("timed_out"):
                return types.ElicitResult(action="decline")
            # Nobody was at the keyboard. Distinct from a decline, and the difference
            # is what the tool tells the model: a cancelled question means stop and
            # ask in the reply, an expired one means carry on with the recommended
            # option. Carried as an accepted result because a decline has no room
            # for it — ``action`` is the only thing a declined result reports.
            return types.ElicitResult(
                action="accept",
                content={
                    "answers": json.dumps([]),
                    "timed_out": True,
                    "timeout_secs": int(result.get("timeout_secs")
                                        or USER_QUESTION_TIMEOUT_SECS),
                },
            )

        # ``ElicitResult.content`` cannot hold a list of answer objects, so the batch travels as
        # one JSON string the interaction server decodes.
        return types.ElicitResult(
            action="accept",
            content={"answers": json.dumps(answers)},
        )

    return _callback


def server_environment(agent: AgentPort) -> dict[str, str]:
    """The environment a server of *agent* is started with, frozen at spawn."""
    # What a server is told at spawn, frozen in its environment: the workspace root, the state
    # dir, the scratchpad (vetted once by the client), MIMIR_SESSION_ID (an agent serves one
    # session for life; empty for the CLI and tests) and MIMIR_DEFAULT_MODEL. ``agent.server_env``
    # goes last and wins: a sub-agent's own session and roots.
    root = getattr(agent, "workspace_root", None) or os.getcwd()
    return {
        **os.environ,
        "MCP_FILES_ROOT": root,
        "SEARCH_ROOT": root,
        "MIMIR_STATE_DIR": paths.STATE_DIR,
        "MIMIR_GLOBAL_STATE_DIR": GLOBAL_STATE_DIR,
        "MIMIR_SCRATCH_DIR": scratch_home(),
        "MIMIR_SESSION_ID": getattr(agent, "session_id", "") or "",
        "MIMIR_DEFAULT_MODEL": agent.model,
        "LLM_BACKEND": getattr(agent, "backend", "") or os.environ.get("LLM_BACKEND", ""),
        **(getattr(agent, "server_env", None) or {}),
    }


async def _spawn_session(*, agent: AgentPort, name: str, script: str) -> Any:
    """Start a server and open its session, short of ``initialize``, on the agent's exit stack.

    Separate so the handshakes of a whole registry overlap.
    """
    if not (script.endswith(".py") or script.endswith(".js")):
        raise ValueError("Server script must be a .py or .js file")

    command = sys.executable if script.endswith(".py") else "node"
    server_env = server_environment(agent)
    params = StdioServerParameters(command=command, args=server_args(script), env=server_env)

    transport = await agent.exit_stack.enter_async_context(stdio_client(params))
    stdio, write = transport
    return await agent.exit_stack.enter_async_context(
        ClientSession(stdio, write, elicitation_callback=_make_elicitation_callback(agent))
    )


async def _register_session(*, agent: AgentPort, name: str, session: Any, listing: Any) -> None:
    """Record an initialized session's tools and resources on the agent.

    Callers register one server at a time, in registry order: the tool schema is sent
    in ``agent.tools`` order on every request, so an order that varies between runs
    moves the prompt prefix and costs the backend's cache.
    """
    agent.sessions[name] = session

    registered: list[str] = []
    for tool in listing.tools:
        owner = agent.tool_owner.get(tool.name)
        if owner is not None and owner != name:
            # First server wins. Re-registered, the name was routed to the second
            # server while the model was shown both schemas under one name.
            logger.warning("tool %r of server %r is already provided by %r; ignored",
                           tool.name, name, owner)
            print(f"⚠️ [{name}]  tool {tool.name!r} already provided by [{owner}] — ignored")
            continue
        registered.append(tool.name)
        agent.tool_owner[tool.name] = name
        # The tool's semantics (capabilities, arg roles, fallbacks, label), from what
        # the server declares (meta/annotations); see tools/capabilities.py.
        agent.tool_caps[tool.name] = infer_tool_caps(tool)
        # Convert MCP tool -> Ollama function-calling format.
        # Parameter descriptions lifted out of the docstring's `Args:` block: a
        # constraint only stated in prose does not get followed. Lifted, the block
        # leaves the description, or every parameter is paid for twice.
        parameters = _schema_for_model(tool)
        agent.tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": _description_without_args_block(
                    tool.description or "", parameters
                ),
                "parameters": parameters,
            },
        })

    tool_names = registered

    n_resources = await register_resources(agent=agent, name=name, session=session)

    suffix = f", {n_resources} resources" if n_resources else ""
    print(f"✅ [{name}]  {len(tool_names)} tools{suffix}: {tool_names}")


async def _handshake(session: Any) -> Any:
    """Complete the MCP handshake on *session* and return its tool listing."""
    await session.initialize()
    return await session.list_tools()


def server_args(script: str) -> list[str]:
    """The arguments that start *script*: a bundled server runs as its module through
    the launcher (``shared/launcher``); anything else by its path."""
    bundled = os.path.realpath(SERVER_BASE)
    path = os.path.realpath(script)
    if script.endswith(".py") and path.startswith(bundled + os.sep):
        module = os.path.relpath(path, bundled)[:-3].replace(os.sep, ".")
        return launch_argv(f"mimir.servers.{module}")
    return [script]


async def connect_server(*, agent: AgentPort, name: str, script: str) -> None:
    """Spawn one MCP server process and register its tools on the agent."""
    session = await _spawn_session(agent=agent, name=name, script=script)
    listing = await _handshake(session)
    await _register_session(agent=agent, name=name, session=session, listing=listing)


async def connect_servers(
    *,
    agent: AgentPort,
    registry: dict[str, str],
    on_error: Callable[[str, BaseException], None] | None = None,
) -> None:
    """Connect every server in *registry*: processes started together, handshakes awaited at once.

    Registration stays in *registry* order, so the tool schema is stable. *on_error* gets
    each failure and the rest still connect; without it the first failure raises.
    """
    spawned: list[tuple[str, Any]] = []
    for name, script in registry.items():
        try:
            spawned.append((name, await _spawn_session(agent=agent, name=name, script=script)))
        except Exception as exc:
            if on_error is None:
                raise
            on_error(name, exc)

    listings = await asyncio.gather(
        *(_handshake(session) for _, session in spawned), return_exceptions=True
    )

    for (name, session), listing in zip(spawned, listings):
        if isinstance(listing, BaseException):
            if on_error is None:
                raise listing
            on_error(name, listing)
            continue
        await _register_session(agent=agent, name=name, session=session, listing=listing)


async def register_resources(*, agent: AgentPort, name: str, session: Any) -> int:
    """Discover a server's MCP resources into ``agent.resources``; return the count (fail-open)."""
    n_resources = 0
    try:
        res_listing = await session.list_resources()
    except Exception:
        logger.debug("register_resources: a best-effort step failed and was skipped", exc_info=True)
        return 0
    for resource in res_listing.resources:
        uri = str(resource.uri)
        agent.resources[uri] = {
            "name": resource.name or uri,
            "description": resource.description or "",
            "mimeType": getattr(resource, "mimeType", None),
            "session": name,
        }
        n_resources += 1
    return n_resources
