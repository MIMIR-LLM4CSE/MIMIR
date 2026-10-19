"""The WebSocket server itself: resolve the model, bind, serve connections, shut down.

Started by ``server.main`` once the command line is applied — the protocol is
documented there.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any, Callable

from ....config import DEFAULT_MODEL
from ....llm.openai_compat import OpenAICompatBackend
from . import registry as server_registry
from ..runtime import CTX_DEFAULT_FULL, ORIGINAL_STDOUT, ensure_router_installed, get_backend
from ..pool.pool import AgentPool
from ..connection.connection import Connection

logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError as exc:
    raise ImportError(
        "websockets is required: pip install websockets"
    ) from exc


__all__ = ["serve"]


# Ceiling on an inbound frame. Sized for the client transcript, the only message that
# can get large: a long session's tool rows, diffs and clipped command output.
_MAX_FRAME_BYTES = 32 * 1024 * 1024
# How long a starting server waits for the lock of one that is stopping: closing every
# agent's MCP servers takes seconds, and a minute covers a slow one.
_LOCK_WAIT_SECS = 60.0


def _announced_address(sockets: Any) -> tuple[str, int]:
    """The host and port behind :func:`_announced_url` — the same choice, unformatted."""
    names = [s.getsockname() for s in sockets]
    addr, port = next((n for n in names if ":" not in n[0]), names[0])[:2]
    return addr, int(port)


def _announced_url(sockets: Any) -> str:
    """The address a client must dial: the literal IPv4 socket when there is one.

    ``localhost`` binds one socket per family, each with its own port under ``--port 0``,
    and ``::1`` is rarely in ``no_proxy``.
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
        backend_obj = get_backend()
        if not isinstance(backend_obj, OpenAICompatBackend):
            return advice
        base_url = backend_obj.endpoint.config()[0]
        _, reason = backend_obj.probe_models()
    except Exception as exc:  # config resolution itself failed
        return f"{advice} (could not resolve the endpoint address: {exc})"
    backend = os.environ.get("LLM_BACKEND", "vllm")
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
    ensure_router_installed()
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
            print(f"Auto-selected served model: {_model}", file=ORIGINAL_STDOUT)
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
            print(f"Model context window: {win:,} tokens (model: {_model})", file=ORIGINAL_STDOUT)
        else:
            print(
                f"WARNING: could not detect context window for '{_model}' — "
                f"falling back to the static {CTX_DEFAULT_FULL:,}-token budget. "
                f"The endpoint's /v1/models may not report max_model_len.",
                file=ORIGINAL_STDOUT,
            )
    except Exception as _e:
        print(f"WARNING: context-window detection failed: {_e!r}", file=ORIGINAL_STDOUT)

    # No port here (with --port 0 the kernel picks it; "Listening on" below says it), and the
    # interpreter line shows what the process really runs on.
    print(f"MIMIR WS server starting on {host}  (model: {_model})", file=ORIGINAL_STDOUT)
    print(f"Python: {sys.executable}  (prefix {sys.prefix})", file=ORIGINAL_STDOUT)
    print("Initialising agent connections…", file=ORIGINAL_STDOUT)

    # The pool: one agent per conversation, built on its first query, shared by every
    # connection. One server per workspace, enforced before binding (see registry.acquire).
    if not await _acquire_lock():
        existing = server_registry.current()
        if existing:
            # Say where the real one is, on the line the extension parses. A caller
            # that meant to spawn then connects to the server that already serves this
            # workspace instead of adding a second.
            print(f"Another MIMIR server already serves this workspace "
                  f"(pid {existing['pid']}). Attaching to it instead of starting a "
                  f"second one.", file=ORIGINAL_STDOUT)
            print(f"Listening on {existing['url']}", file=ORIGINAL_STDOUT, flush=True)
            return
        print("Another process holds this workspace's server lock but published no "
              "address. Refusing to start a second server: that would corrupt the "
              "shared journal. Stop the other server, or clear "
              f"{server_registry.lock_path()} if nothing is running.",
              file=ORIGINAL_STDOUT, flush=True)
        raise SystemExit(1)

    pool = AgentPool(_model)
    # The event pump, started here rather than with the first worker: it belongs to the
    # process, and what it does — drain every worker, journal what they emit, hand it to
    # whatever sockets are attached — has to happen when none are.
    pool.ensure_pump()
    print(f"Agent pool ready (up to {pool.cap} live conversations).",
          file=ORIGINAL_STDOUT)

    async def _handler(ws: Any) -> None:
        session = Connection(ws, pool)
        await session.run()

    # The default 1 MiB frame cap is below what a client transcript weighs once it
    # carries diffs and command output, and an oversized frame closes the connection
    # rather than failing the one message.
    async with websockets.serve(_handler, host, port, max_size=_MAX_FRAME_BYTES) as server:
        # Read the port back off the socket rather than echoing the argument: with
        # --port 0 the argument is a placeholder, and this line is the contract the
        # VS Code extension parses to learn where to connect.
        url = _announced_url(server.sockets)
        print(f"Listening on {url}", file=ORIGINAL_STDOUT, flush=True)
        # Written down as well as announced. The printed line reaches whoever holds this
        # process's stdout — the extension host that spawned it, and nobody else; a
        # server meant to outlive that window has to leave its address somewhere a
        # later window can read. One file per workspace, which is where "one server per
        # workspace" actually comes from.
        _addr = _announced_address(server.sockets)
        server_registry.publish(url=url, host=_addr[0], port=_addr[1],
                                model=pool.model)
        def _retire() -> None:
            # First thing on the way out, before the agents are closed: closing them
            # takes seconds, and a window opened meanwhile must not find this address and
            # attach to a server that is going away. Listening stops with it; the
            # connections already open finish on their own.
            server_registry.clear()
            server.close(close_connections=False)

        try:
            await _run_until_signalled(pool, retire=_retire)
        finally:
            # Retired here rather than in the signal path: a crash leaves it behind on
            # purpose, and a reader checks liveness instead of trusting the file.
            server_registry.clear()
            server_registry.release()


async def _acquire_lock() -> bool:
    """Take the workspace's server lock, waiting while its holder is on the way out.

    A stopping server holds the lock while it closes its agents, its address already
    withdrawn, so a server started then waits rather than refuse. False once a live
    server answers, or when the wait ends with the lock still held.
    """
    deadline = asyncio.get_running_loop().time() + _LOCK_WAIT_SECS
    while True:
        if server_registry.acquire():
            return True
        if server_registry.current() is not None:
            return False
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.25)


async def _run_until_signalled(pool: AgentPool, retire: Callable[[], None] = lambda: None) -> None:
    """Serve until a signal, a client ``shutdown`` or the pool's idle stop; then close every agent.

    *retire* runs first, so nothing new reaches a server that is stopping. Only closing an
    agent's exit stack reaps its MCP servers (each in its own process group). Uses
    ``add_signal_handler``; without it (Windows) serves until stopped.
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
    pool.lifecycle.stop_requested = asyncio.Event()
    asked = loop.create_task(pool.lifecycle.stop_requested.wait())

    installed = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _ask_to_stop)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    try:
        await asyncio.wait([stop, asked], return_when=asyncio.FIRST_COMPLETED)
        why = "idle" if pool.lifecycle.stop_requested.is_set() else "asked to"
        print(f"Stopping ({why}) — closing agent connections…",
              file=ORIGINAL_STDOUT, flush=True)
    finally:
        try:
            retire()
        except Exception:
            logger.warning("stopping: could not retire this server's address", exc_info=True)
        asked.cancel()
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        await pool.aclose_all()
