from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import inspect
import json
import os
import re
import sys
from collections.abc import Callable
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .. import human_pause
from ..context.capabilities import infer_tool_caps
from ..tool_execution.tool_status_messages import DOING_ARG
from ..config.constants import GLOBAL_STATE_DIR, STATE_DIR, USER_QUESTION_TIMEOUT_SECS
from ...servers._shared.state_paths import scratch_home


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
    """The ``Args:`` block of *doc*, entry by entry, with its line bounds.

    Returns ``(entries, first, end)``: each entry is ``(names, text, start, stop)``,
    and every bound is a line index, ``first`` on the header and the ends excluded,
    so extracting the entries and cutting them out agree on where each one lies.
    None when there is no block.
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

    The block is lifted into the parameters (:func:`_schema_for_model`),
    and it was also left in the description: every parameter was sent twice, on every
    call. On one deployment that was a fifth of a ~31k-token tools schema.

    Cut entry by entry, and only what is not lost: an entry goes when every parameter
    it names has exactly its text in the schema. One whose parameter the schema lacks,
    or whose hand-written ``Field`` description says something else, stays, and the
    header with it. Never raises; on anything unexpected the doc is unchanged.
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
        return doc


# The one argument every tool takes, added to each schema below rather than declared by
# 60-odd servers. It is the model's own sentence about *this* call — what the user reads
# beside the tool's family while it runs — and the only part of a row that knows why the
# call is happening. The dispatcher strips it before the call is executed (see
# ``query_engine.dispatch``): no server ever receives it.
#
# Deliberately never added to ``required``: a backend that validates the schema would
# refuse a call that omits it, and a row without a description is a case the UI already
# renders (its family and its target still say what it touched).
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
    """The tool's input schema as the model sees it: ``doing`` added, ``Args:`` lifted.

    Both changes are here for the same reason — the schema is the part that gets read.
    ``doing`` is the universal per-call description (see :data:`DOING_ARG`), added to
    every tool rather than declared sixty times.

    As for the rest: the constraint a tool documents only in prose does not change what the model does.
    Observed on a real run: the model dropped `report_verdict`'s required `verdict`
    twice, called `proxy_get` with `op="help"` (not one of the six the docstring lists)
    and with `op="report"` while the same docstring says "requires: name", and hit
    `bash_run`'s 30s default four times against a docstring that says to raise it. The
    prose reaches the model — the whole docstring, ``Args:`` block included, is the
    tool's `description` — and was not followed. What the schema says is what gets
    obeyed, and every parameter here was a bare ``{"type": "string"}``.

    So the text is not rewritten, it is *moved to where it is read*: the ``Args:``
    blocks are already written and maintained beside the code they describe, and this
    lifts them into the schema. Descriptions already set — by an explicit
    ``Field(description=...)`` on the server — always win: a hand-written one is a
    deliberate act, and this is a fallback for the ones nobody got to.

    Returns a deep copy. ``infer_tool_caps`` reads ``tool.inputSchema`` too, and it
    must keep seeing exactly what the server declared.
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
        return schema
    for name, prop in props.items():
        if isinstance(prop, dict) and not prop.get("description") and described.get(name):
            prop["description"] = described[name]
    return schema


def _bind_question_handler(ask: Any):
    """``ask`` with the answer deadline applied, as a one-argument callable.

    Frontends take the wall as a second parameter; handlers that do not offer one —
    the non-interactive default's replacements in tests, a third-party shim — are
    called as they always were and simply wait as long as they used to.
    """
    try:
        inspect.signature(ask).bind(["q"], timeout_secs=USER_QUESTION_TIMEOUT_SECS)
    except (TypeError, ValueError):
        return ask
    return functools.partial(ask, timeout_secs=USER_QUESTION_TIMEOUT_SECS)


def _make_elicitation_callback(agent: Any):
    """Bridge MCP elicitation requests to the active frontend.

    A server tool (e.g. ``ask_user_question``) calls ``ctx.session.elicit_form``;
    that arrives here as an ``elicitation/create`` request. We parse the rich
    question spec carried in the schema's ``x_mimir`` extension and hand it to
    ``agent._request_user_question`` — a per-frontend handler (CLI ``input``, a
    WebSocket prompt card, or the default no-op) that asks the (possibly several)
    questions sequentially and returns the user's answers.

    The frontend handlers are blocking/queue-based, so we run them off the event
    loop via ``run_in_executor`` to keep the agent loop responsive. The wait is
    marked as human time *here*, on the loop thread, and not only inside the
    handler: ``human_pause`` accounting is thread-local, so a pause recorded on a
    pool thread is invisible to the ``_await_tool`` that is watching this tool call
    from the loop thread, and ``ask_user_question`` died on the ordinary tool budget
    after two minutes while the card on screen said five.

    The handler is given ``USER_QUESTION_TIMEOUT_SECS`` as the wall to answer within.
    The wall belongs to this path and not to the handler itself: plan approval calls
    the same handler and waits indefinitely, which is why the timeout travels as an
    argument instead of being read from the constant inside each frontend.
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

        loop = asyncio.get_running_loop()
        try:
            # With the caller's context: the handler reads which tool call is asking
            # (query_engine.deferral), and a bare executor thread starts from none.
            with human_pause.human_pause():
                result = await loop.run_in_executor(
                    None,
                    contextvars.copy_context().run,
                    _bind_question_handler(agent._request_user_question),
                    questions,
                )
        except Exception:
            return types.ElicitResult(action="cancel")

        result = result or {}
        answers = list(result.get("answers") or [])
        if not answers:
            if not result.get("timed_out"):
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

        # ``ElicitResult.content`` is typed by the MCP SDK as
        # ``dict[str, str | int | float | bool | list[str] | None]`` — a list of
        # per-question answer objects does not fit it, and pydantic rejects the
        # construction outright. So the batch travels as one JSON string and the
        # interaction server decodes it. Sending the raw list made *answering*
        # fail while declining succeeded: the validation error surfaced to the
        # tool as "could not ask the user", and the run silently proceeded on the
        # model's own judgment with the user's actual answer discarded.
        return types.ElicitResult(
            action="accept",
            content={"answers": json.dumps(answers)},
        )

    return _callback


async def _spawn_session(*, agent: Any, name: str, script: str) -> Any:
    """Start a server process and open its session, stopping short of ``initialize``.

    Split from the registration half so a whole registry can be started at once:
    launching the process is immediate, while the handshake waits on the child's
    interpreter importing the MCP SDK — two thirds of a second, the same two thirds
    in every one of them. Serialized that is the startup; overlapped it is paid once.
    Every session is entered on the agent's own exit stack from the caller's task,
    so the stack stays single-task and ``stdio_client``'s cancel scope is unwound
    where it was opened.
    """
    if not (script.endswith(".py") or script.endswith(".js")):
        raise ValueError("Server script must be a .py or .js file")

    command = sys.executable if script.endswith(".py") else "node"
    # Propagate the client's cwd as the file-server root so paths are predictable,
    # and the central per-workspace state dir so state-owning servers (memory, todo,
    # platform profile) persist to the same place the client resolves. The state dir
    # is deliberately distinct from the file/search sandbox — see config.constants.
    # The scratchpad travels the same way: the client vetted that path at startup
    # (ensure_scratch_home), so servers must use its answer, not re-derive one.
    #
    # MIMIR_SESSION_ID is what makes a server's state per-conversation. One agent serves
    # one session for its whole life, so the session is fixed at spawn and the frozen
    # environment is the right place for it: every state path the server resolves
    # (todo list, plans, approved paths, run channels, job dirs) then names its own
    # session, whatever other sessions are doing at the same moment. Empty for the ends
    # that have no session — the CLI, standalone runs, tests — which fall back to the
    # active-session pointer.
    #
    # MIMIR_DEFAULT_MODEL travels per server rather than through the client's own
    # environment: server_spawn_agent reads it out of this frozen copy to build its
    # child, and a second session on another model must not change what an
    # already-spawned server would pass on.
    server_env = {
        **os.environ,
        "MCP_FILES_ROOT": os.getcwd(),
        "SEARCH_ROOT": os.getcwd(),
        "MIMIR_STATE_DIR": STATE_DIR,
        "MIMIR_GLOBAL_STATE_DIR": GLOBAL_STATE_DIR,
        "MIMIR_SCRATCH_DIR": scratch_home(),
        "MIMIR_SESSION_ID": getattr(agent, "session_id", "") or "",
        "MIMIR_DEFAULT_MODEL": getattr(agent, "model", "") or "",
        "LLM_BACKEND": getattr(agent, "backend", "") or os.environ.get("LLM_BACKEND", ""),
    }
    params = StdioServerParameters(command=command, args=[script], env=server_env)

    transport = await agent.exit_stack.enter_async_context(stdio_client(params))
    stdio, write = transport
    return await agent.exit_stack.enter_async_context(
        ClientSession(stdio, write, elicitation_callback=_make_elicitation_callback(agent))
    )


async def _register_session(*, agent: Any, name: str, session: Any, listing: Any) -> None:
    """Record an initialized session's tools and resources on the agent.

    Callers register one server at a time, in registry order: the tool schema is sent
    in ``agent.tools`` order on every request, so an order that varies between runs
    moves the prompt prefix and costs the backend's cache.
    """
    agent.sessions[name] = session

    for tool in listing.tools:
        agent.tool_owner[tool.name] = name
        # Derive the tool's semantics (capabilities, arg roles, fallbacks, label)
        # from what the server declares (meta/annotations), falling back to the
        # relocated legacy table.  This registry replaces the old scattered
        # hardcoded tool-name lists; see context/capabilities.py.
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

    tool_names = [t.name for t in listing.tools]

    n_resources = await register_resources(agent=agent, name=name, session=session)

    suffix = f", {n_resources} resources" if n_resources else ""
    print(f"✅ [{name}]  {len(tool_names)} tools{suffix}: {tool_names}")


async def _handshake(session: Any) -> Any:
    """Complete the MCP handshake on *session* and return its tool listing."""
    await session.initialize()
    return await session.list_tools()


async def connect_server(*, agent: Any, name: str, script: str) -> None:
    """Spawn one MCP server process and register its tools on the agent."""
    session = await _spawn_session(agent=agent, name=name, script=script)
    listing = await _handshake(session)
    await _register_session(agent=agent, name=name, session=session, listing=listing)


async def connect_servers(
    *,
    agent: Any,
    registry: dict[str, str],
    on_error: Callable[[str, BaseException], None] | None = None,
) -> None:
    """Connect every server in *registry*, overlapping the handshakes.

    Start to answering a first query, the registry is the wait: a score of child
    interpreters each importing the same SDK, one after the next. The processes are
    started together instead and the handshakes awaited at once, which puts the whole
    registry at the cost of its slowest member rather than the sum of all of them.

    Registration stays sequential and in *registry* order, so the tool schema the
    model is sent does not depend on which child answered first.

    *on_error* receives the name and the failure of a server that could not be
    connected, and the rest still connect — that is how a sub-agent tolerates a
    server it does not need. With no handler the first failure is raised, leaving
    the caller to close the agent.
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


async def register_resources(*, agent: Any, name: str, session: Any) -> int:
    """Discover a server's MCP resources into ``agent.resources``; return the count.

    Resources are user-attached context (Claude/Copilot-style ``@``-mention), not
    model-invokable tools — so they go into a separate registry, never into
    ``agent.tools``. Servers that don't implement the capability raise or return an
    empty list; either is treated as "no resources" (fail-open). URIs are stringified
    (they arrive as pydantic ``AnyUrl``).
    """
    n_resources = 0
    try:
        res_listing = await session.list_resources()
    except Exception:
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
