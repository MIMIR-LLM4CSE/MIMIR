"""WebSocket server for MimirAgent.

Provides a streaming JSON protocol consumed by the VS Code extension (or any
WebSocket client).  The MimirAgent runs in a dedicated background thread with its
own asyncio event loop so blocking approval prompts never freeze the WebSocket
event loop.

This module is the entry point (``serve`` / ``main``); the implementation is split
across sibling modules:
  - ``_ws_runtime`` — shared foundation (cwd bootstrap, stdout router, todo helpers,
    context-budget constants).
  - ``ws_worker``   — ``_AgentWorker``, the background agent thread.
  - ``ws_pool``     — ``_AgentPool``, one worker per conversation: built on that
                      conversation's first query, released when it goes idle.
  - ``ws_session``  — ``_Session``, one WebSocket connection.
``_AgentWorker``, ``_AgentPool`` and ``_Session`` are re-exported here for
backward compatibility.

Protocol — all messages are JSON objects, one per send/recv:

  Server → Client
    {"type": "ready",          "model": "...", "context_mode": "...", "enforcement": "...",
                               "thinking": {"mechanism": "kwarg|directive|effort",
                                            "levels": [...], "can_disable": bool},
                               "temperature": {"supported": bool, "value": float|null}}
    {"type": "output",         "text": "..."}          # stdout (tool status + LLM tokens)
    {"type": "enforcement",    "mode": "strict"|"light"|"off"}  # active guidance-nudge level
    {"type": "mode",           "mode": "agent"|"plan"|"ask"}    # server-driven mode switch
                                                                # (plan approval → agent),
                                                                # and the answer to /mode
    {"type": "thinking_depth", "depth": 0-5, "label": "..."}    # the depth the agent HOLDS,
                                                                # after /thinking[-depth]
    {"type": "streaming",      "enabled": bool}                 # after /streaming
    {"type": "temperature",    "supported": bool, "value": float|null}
                                                                # after /temperature; null =
                                                                # the model's own (none sent)
    {"type": "command_output", "command": "/memory list", "title": "3 memories",
                               "items": [{"label": "...", "detail": "..."}],
                               "note": "...", "tone": "ok"|"warn"|"empty"}
                                                                # answer to a session command;
                                                                # settings that own a control
                                                                # report state instead
    {"type": "approval",       "id": "...", "tool": "...", "server": "...",
                               "args": {}, "risk": "...", "scope": "...",
                               "session_id": "...", "session_title": "..."}
    {"type": "user_question",  "id": "...", "questions": [
                               {"question": "...", "header": "...", "multiSelect": false,
                                "options": [{"label": "...", "description": "..."}]}],
                               "session_id": "...", "session_title": "..."}
                                                       # Every card says which
                                                       # conversation raised it. Several
                                                       # run turns at once, so one may ask
                                                       # while the user reads another —
                                                       # and the answer must come back
                                                       # with that id (below).
    {"type": "prompt_expired", "id": "...", "kind": "user_question",
                               "timeout_secs": 300,
                               "session_id": "...", "session_title": "..."}
                                                       # the wait behind that card gave
                                                       # up: close it, here or in the
                                                       # foreign-prompt strip. The turn
                                                       # is producing again — it went on
                                                       # with the option it recommended
    {"type": "queued",         "session_id": "...", "position": 1, "text": "..."}
                                                       # every agent slot is taken; this
                                                       # conversation's turn starts when
                                                       # one frees. Rendered, not dropped:
                                                       # an invisible queue reads as a hang
    {"type": "tool_progress",  "id": "...", "phase": "...", "percent": 0.0}
                                                       # what a blocking run is doing,
                                                       # read off its run channel once a
                                                       # second. Transient: never
                                                       # recorded in the transcript.
    {"type": "tool_backgrounded", "id": "...", "job_key": "..."}
                                                       # a watcher took this call's run;
                                                       # the row settles but the work
                                                       # goes on under that key
    {"type": "job_progress",   "job_key": "...", "phase": "...", "percent": 0.0}
                                                       # the same, for a run already
                                                       # detached — from the watcher
                                                       # that polls it
    {"type": "job_checkin",    "jobs": [{"job_key": "...", "state": "running",
                               "phase": "...", "percent": 0.0}],
                               "resumes_active_session": false}
                                                       # a bulletin on runs still going,
                                                       # 30s / 2min / 10min after the
                                                       # first launch. Settles no row:
                                                       # the news is that they are still
                                                       # there
    {"type": "job_complete",   "job_key": "...", "state": "done", "summary": {...},
                               "resumes_active_session": false}
                                                       # a detached run reached a
                                                       # terminal state. The flag says a
                                                       # turn is starting in THIS
                                                       # conversation, which the client
                                                       # cannot work out for itself
    {"type": "answer",         "text": "..."}          # final answer for a query
    {"type": "todo",           "items": [{"text": "...", "done": false}]}
    {"type": "error",          "text": "..."}
    {"type": "sessions_list",  "sessions": [{"id": "...", "title": "...",
                               "created_at": "...", "updated_at": "...", "preview": "...",
                               "running": false,   # a turn of it is in flight
                               "parked": false,    # its turn waits on a card: no timeout,
                                                   # so it stays stopped until answered
                               "queued": false,    # waiting for an agent slot
                               "runs": 0}]}        # background runs still going, which
                                                   # a conversation has with no turn
                                                   # in flight — the detach case
    {"type": "session_loaded", "session_id": "...", "title": "...",
                               "display_messages": [...], "todos": [...]}
    {"type": "shutting_down",  "forced": false, "reasons": [...]}
                                                       # the server is going. ``reasons``
                                                       # is what was still running, which
                                                       # a forced stop is ending
    {"type": "shutdown_refused", "reasons": [...]}     # it is still needed, and why
    {"type": "detached",       "detached": true, "log": "...",
                               "autonomy": "manual|auto|auto_all",
                               "sessions": [...], "pid": 0, "setsid": true}
                                                       # the answer to ``detach``: this
                                                       # server has re-pointed its output
                                                       # at ``log`` and left the
                                                       # extension host's process group,
                                                       # so it survives the window. The
                                                       # client must stop killing it
    {"type": "replay",         "session_id": "...", "events": [...],
                               "through_seq": 0, "more": false, "truncated": false}
                                                       # what this conversation produced
                                                       # while nobody was attached, read
                                                       # back from its journal past the
                                                       # client's watermark. Sent on
                                                       # attach, in frames; feed each
                                                       # event to the same reducer a live
                                                       # one goes to. ``through_seq`` is
                                                       # the watermark to send back on
                                                       # the next ``transcript``;
                                                       # ``truncated`` means older events
                                                       # were elided. Streamed deltas are
                                                       # not replayed — a resumed turn
                                                       # arrives as the aggregates that
                                                       # closed its blocks
    {"type": "context_usage",  "used_tokens": 0, "total_tokens": 0, "reserved_tokens": 0,
                               "overhead_tokens": 0,
                               "overhead_measured": false,  # true once server-reported
                               "history_messages": 0,        # in the window the model sees
                               "history_messages_full": 0,   # in the untrimmed record
                               "provisional": false}         # no agent yet: the fixed
                                                             # part is not counted in
                                                             # used_tokens, so the figure
                                                             # is a floor, not a verdict
    {"type": "resources",      "resources": [{"uri": "...", "name": "...",
                               "description": "...", "mimeType": "..."}]}  # attachable resources

  Client → Server
    {"type": "query",             "text": "..."}   # @<uri> mentions are read & injected
    {"type": "transcript",        "session_id": "...", "messages": [...]}
                                  # the client's rendered chat, stored verbatim as the
                                  # session's display messages (see _handle_transcript)
    {"type": "list_resources"}                     # request the attachable-resource list
    {"type": "approval_response", "id": "...", "session_id": "...",
                               "choice": "y"|"n"|"a"}
    {"type": "user_question_response", "id": "...", "session_id": "...", "answers": [
                               {"selected": ["..."], "otherText": "..."}]}
                                  # session_id names the conversation that asked, copied
                                  # off the card. An answer without one is DROPPED rather
                                  # than given to whichever conversation is on screen:
                                  # that would settle a question another one asked, with
                                  # the user's approval on a call they never saw. An
                                  # answer to a card whose wait expired is dropped too.
    {"type": "shutdown", "force": false}             # stop the server. Refused while
                                                     # anything is still working, with
                                                     # the reasons, unless forced
    {"type": "detach", "autonomy": "manual|auto|auto_all",
                       "session_ids": ["..."],
                       "enabled": true}             # "continue without me": make this
                                                     # server survivable and set what it
                                                     # may do unattended. No session_ids
                                                     # means every conversation, and only
                                                     # that form outlives a worker rebuild.
                                                     # ``enabled: false`` takes it back:
                                                     # the window owns the server again
    {"type": "divert_to_background", "id": "..."}   # detach the run now blocking the
                                  # turn, keeping what it has already done. The id names
                                  # the row, whose tool name is the run channel its
                                  # server publishes under. Served on the WS loop, never
                                  # through the model: the agent is parked awaiting that
                                  # very call.
    {"type": "command",           "text": "/mode agent|plan|ask"}
    {"type": "create_session"}
    {"type": "switch_session",    "session_id": "..."}
    {"type": "delete_session",    "session_id": "..."}
    {"type": "rename_session",    "session_id": "...", "title": "..."}
"""

from __future__ import annotations

# Import the shared runtime FIRST so its cwd bootstrap runs before config.constants
# (and the backend factory) capture the workspace root at import time.
from ._ws_runtime import (
    _CTX_DEFAULT_FULL,
    _ORIGINAL_STDOUT,
    _ensure_router_installed,
    get_backend,
)
from .ws_worker import _AgentWorker
from .ws_pool import _AgentPool
from . import server_registry
from .ws_session import _Session

import asyncio
import os
import signal
import sys
from typing import Any

try:
    import websockets
except ImportError as exc:
    raise ImportError(
        "websockets is required: pip install websockets"
    ) from exc


__all__ = ["serve", "main", "_AgentWorker", "_AgentPool", "_Session"]


# Ceiling on an inbound frame. Sized for the client transcript, the only message that
# can get large: a long session's tool rows, diffs and clipped command output.
_MAX_FRAME_BYTES = 32 * 1024 * 1024


def _announced_address(sockets: Any) -> tuple[str, int]:
    """The host and port behind :func:`_announced_url` — the same choice, unformatted."""
    names = [s.getsockname() for s in sockets]
    addr, port = next((n for n in names if ":" not in n[0]), names[0])[:2]
    return addr, int(port)


def _announced_url(sockets: Any) -> str:
    """The address a client must dial to reach the server.

    ``localhost`` resolves to both 127.0.0.1 and ::1, so the server binds one socket
    per family — and with ``--port 0`` each gets its own port. The order of the two
    varies between launches. Announcing ``localhost`` with the port of the first
    socket sent the client, which picks 127.0.0.1, to the IPv6 port about half the
    time: refused, on every retry. The literal address of a socket removes the
    guess.

    The IPv4 socket is named when there is one. ``no_proxy`` lists ``127.0.0.1`` and
    ``localhost`` almost everywhere, but rarely ``::1``: a client that honours the
    proxy variables sent ``ws://[::1]`` to the corporate proxy, which closed it.
    """
    addr, port = _announced_address(sockets)
    host = f"[{addr}]" if ":" in addr else addr
    return f"ws://{host}:{port}"


def _no_model_message() -> str:
    """Explain why no model could be resolved, naming the endpoint that was asked.

    Falls back to the plain "pass --model" advice for backends that cannot
    enumerate themselves (Ollama, Anthropic), where an empty list says nothing
    about reachability.
    """
    advice = "No model specified. Pass --model <name> or set MIMIR_DEFAULT_MODEL."
    try:
        from ...query_engine.backends.vllm_backend import probe_models, _get_vllm_config
    except ImportError:
        return advice
    backend = os.environ.get("LLM_BACKEND", "vllm")
    if backend not in ("vllm", "ray"):
        return advice
    try:
        base_url, api_key = _get_vllm_config()
        _, reason = probe_models((base_url, api_key))
    except Exception as exc:  # config resolution itself failed
        return f"{advice} (could not resolve the endpoint address: {exc})"
    if reason:
        return (
            f"The {backend} endpoint at {base_url} did not answer, so the served "
            f"model could not be resolved ({reason}). Check that it is running and "
            f"reachable from here, then connect again."
        )
    return (
        f"The {backend} endpoint at {base_url} answered but serves no model. "
        f"{advice}"
    )


async def serve(
    host: str = "localhost",
    port: int = 8765,
    model: str | None = None,
) -> None:
    """Start the WebSocket server (runs forever)."""
    try:
        from ...config import DEFAULT_MODEL
    except ImportError:
        from mimir.client.config import DEFAULT_MODEL

    _ensure_router_installed()
    # Model resolution: explicit arg > MIMIR_DEFAULT_MODEL env var > DEFAULT_MODEL config
    _model = model or os.environ.get("MIMIR_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL
    if not _model:
        # "Connect to running server" mode: no model was picked, so use whatever the
        # endpoint is already serving. Backends that cannot enumerate themselves
        # (Ollama, Anthropic) answer with an empty list, so this needs no test on
        # which backend is active.
        served = get_backend().served_models()
        if served:
            _model = served[0]
            print(f"Auto-selected served model: {_model}", file=_ORIGINAL_STDOUT)
    if not _model:
        # An unreachable endpoint and one serving nothing both leave the list
        # empty, and blaming the missing --model sends the user hunting for a
        # setting when the real answer is that nothing answered at that address.
        raise ValueError(_no_model_message())

    # Prime the context-window cache (/v1/models for vLLM and Ray, /api/show for
    # Ollama) so the budget checks on the WS event loop hit the cache instead of
    # blocking.
    try:
        win = get_backend().context_window(_model)
        if win:
            print(f"Model context window: {win:,} tokens (model: {_model})", file=_ORIGINAL_STDOUT)
        else:
            print(
                f"WARNING: could not detect context window for '{_model}' — "
                f"falling back to the static {_CTX_DEFAULT_FULL:,}-token budget. "
                f"The endpoint's /v1/models may not report max_model_len.",
                file=_ORIGINAL_STDOUT,
            )
    except Exception as _e:
        print(f"WARNING: context-window detection failed: {_e!r}", file=_ORIGINAL_STDOUT)

    # The port is deliberately absent here: with --port 0 (what the VS Code extension
    # passes, so each window gets its own server) it is the kernel that picks one, and
    # it is not known until the socket is bound. The "Listening on" line below is the
    # one that carries the real address.
    # The interpreter line is the server's own view of what it runs on: the extension
    # logs the binary it launched, but a stray PYTHONHOME/PYTHONPATH in the inherited
    # environment can still bend that interpreter onto another installation — this is
    # the line that says so, from inside the process, where it cannot be mistaken.
    print(f"MIMIR WS server starting on {host}  (model: {_model})", file=_ORIGINAL_STDOUT)
    print(f"Python: {sys.executable}  (prefix {sys.prefix})", file=_ORIGINAL_STDOUT)
    print("Initialising agent connections…", file=_ORIGINAL_STDOUT)

    # The pool, not an agent: one agent per conversation, built on that conversation's
    # first query. Process-global and shared by every connection, because two webviews on
    # one port must see the same live turns — which is what the single shared worker gave
    # for free, and the one property of it worth keeping.
    #
    # Nothing is built here any more, so the socket is listening in milliseconds instead
    # of after the backend wait plus ~19 server spawns. That cost did not disappear; it
    # moved to the first query of each conversation, which is the only place that can say
    # which session is paying it.
    # One server per workspace, enforced before anything is bound. Two of them share
    # the sessions directory, so both append to the same journal and both derive `seq`
    # from it: the numbering collides, the watermark built on it stops meaning
    # anything, and a client attached to one sees nothing of the turn running in the
    # other — a conversation whose tools run and never appear.
    if not server_registry.acquire():
        existing = server_registry.current()
        if existing:
            # Say where the real one is, on the line the extension parses. A caller
            # that meant to spawn then connects to the server that already serves this
            # workspace instead of adding a second.
            print(f"Another MIMIR server already serves this workspace "
                  f"(pid {existing['pid']}). Attaching to it instead of starting a "
                  f"second one.", file=_ORIGINAL_STDOUT)
            print(f"Listening on {existing['url']}", file=_ORIGINAL_STDOUT, flush=True)
            return
        print("Another process holds this workspace's server lock but published no "
              "address. Refusing to start a second server: that would corrupt the "
              "shared journal. Stop the other server, or clear "
              f"{server_registry.lock_path()} if nothing is running.",
              file=_ORIGINAL_STDOUT, flush=True)
        raise SystemExit(1)

    pool = _AgentPool(_model)
    # The event pump, started here rather than with the first worker: it belongs to the
    # process, and what it does — drain every worker, journal what they emit, hand it to
    # whatever sockets are attached — has to happen when none are.
    pool.ensure_pump()
    print(f"Agent pool ready (up to {pool.cap} live conversations).",
          file=_ORIGINAL_STDOUT)

    async def _handler(ws: Any) -> None:
        session = _Session(ws, pool)
        await session.run()

    # The default 1 MiB frame cap is below what a client transcript weighs once it
    # carries diffs and command output, and an oversized frame closes the connection
    # rather than failing the one message.
    async with websockets.serve(_handler, host, port, max_size=_MAX_FRAME_BYTES) as server:
        # Read the port back off the socket rather than echoing the argument: with
        # --port 0 the argument is a placeholder, and this line is the contract the
        # VS Code extension parses to learn where to connect.
        url = _announced_url(server.sockets)
        print(f"Listening on {url}", file=_ORIGINAL_STDOUT, flush=True)
        # Written down as well as announced. The printed line reaches whoever holds this
        # process's stdout — the extension host that spawned it, and nobody else; a
        # server meant to outlive that window has to leave its address somewhere a
        # later window can read. One file per workspace, which is where "one server per
        # workspace" actually comes from.
        _addr = _announced_address(server.sockets)
        server_registry.publish(url=url, host=_addr[0], port=_addr[1],
                                model=pool.model)
        try:
            await _run_until_signalled(pool)
        finally:
            # Retired here rather than in the signal path: a crash leaves it behind on
            # purpose, and a reader checks liveness instead of trusting the file.
            server_registry.clear()
            server_registry.release()


async def _run_until_signalled(pool: _AgentPool) -> None:
    """Serve until nothing needs this process, then close every agent's MCP servers.

    ``stdio_client`` spawns each server with ``start_new_session=True`` — its own process
    group, so it survives this process dying — and the only thing that terminates one is
    closing the agent's exit stack. Dying without that leaves up to ``pool.cap`` × ~19
    orphaned interpreters behind, each holding whatever its own child processes hold; the
    VS Code extension kills and respawns this server on every connect, so that is the
    ordinary path, not an edge case.

    ``add_signal_handler`` rather than ``signal.signal``: the close runs on this loop, and
    a handler that interrupts an arbitrary frame cannot await. Falls back to serving
    forever where the loop does not support it (Windows), which is where the pool's own
    idle release is the only reaping there is.

    Three doors, one exit. A signal, an explicit ``shutdown`` from the client, and the
    pool deciding it is no longer needed all resolve here, because what has to happen on
    the way out — unwinding each agent's exit stack — is the same in every case and is
    the only thing that reaps the MCP servers.
    """
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def _ask_to_stop() -> None:
        if not stop.done():
            stop.set_result(None)

    # The third door out, beside SIGTERM and SIGINT: the pool deciding nothing needs
    # this process any more. It goes through the same exit precisely because that exit
    # is the only thing that closes the MCP servers — a stop that took a shortcut would
    # leave the orphans this function exists to prevent.
    pool.stop_requested = asyncio.Event()
    asked = loop.create_task(pool.stop_requested.wait())

    installed = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _ask_to_stop)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    try:
        await asyncio.wait([stop, asked], return_when=asyncio.FIRST_COMPLETED)
        why = "idle" if pool.stop_requested.is_set() else "asked to"
        print(f"Stopping ({why}) — closing agent connections…",
              file=_ORIGINAL_STDOUT, flush=True)
    finally:
        asked.cancel()
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        await pool.aclose_all()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MIMIR WebSocket server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to bind. 0 lets the OS pick a free one — the chosen "
                             "port is then printed on the 'Listening on ws://…' line.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cwd", default=None, help="Set working directory before starting")
    parser.add_argument("--backend", choices=["ollama", "vllm", "ray", "anthropic"], default=None,
                        help="LLM backend to use (default: from LLM_BACKEND env var or vllm). "
                             "ray is a Ray Serve LLM router — the same OpenAI API as vLLM, at "
                             "its own address. anthropic uses the hosted Claude API — the key "
                             "comes from the ANTHROPIC_API_KEY env var, never a CLI arg.")
    parser.add_argument("--vllm-base-url", default=None,
                        help="Base URL of the vLLM OpenAI-compatible API (overrides VLLM_BASE_URL env var)")
    parser.add_argument("--vllm-api-key", default=None,
                        help="API key for vLLM (default: EMPTY)")
    parser.add_argument("--ray-base-url", default=None,
                        help="Base URL of the Ray Serve LLM router, including the app's route "
                             "prefix if it has one (overrides RAY_BASE_URL env var)")
    parser.add_argument("--ray-api-key", default=None,
                        help="API key for the Ray Serve router (default: EMPTY)")
    parser.add_argument("--ollama-base-url", default=None,
                        help="Base URL of the running Ollama server "
                             "(overrides OLLAMA_BASE_URL / OLLAMA_HOST)")
    args = parser.parse_args()

    if args.cwd:
        os.chdir(args.cwd)

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    if args.vllm_api_key:
        os.environ["VLLM_API_KEY"] = args.vllm_api_key
    if args.ray_base_url:
        os.environ["RAY_BASE_URL"] = args.ray_base_url
    if args.ray_api_key:
        os.environ["RAY_API_KEY"] = args.ray_api_key
    if args.ollama_base_url:
        # Two names, one address: OLLAMA_BASE_URL is what the health check and the
        # model pre-warm read, OLLAMA_HOST is what the `ollama` package resolves
        # its client from. Setting only one leaves half the process pointing at
        # the default localhost.
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url
        os.environ["OLLAMA_HOST"] = args.ollama_base_url

    asyncio.run(serve(host=args.host, port=args.port, model=args.model))


if __name__ == "__main__":
    import pathlib as _pathlib
    _repo_root = str(_pathlib.Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    main()
