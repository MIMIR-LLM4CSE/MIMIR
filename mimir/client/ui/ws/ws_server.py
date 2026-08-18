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
  - ``ws_session``  — ``_Session``, one WebSocket connection.
``_AgentWorker`` and ``_Session`` are re-exported here for backward compatibility.

Protocol — all messages are JSON objects, one per send/recv:

  Server → Client
    {"type": "ready",          "model": "...", "context_mode": "...", "enforcement": "..."}
    {"type": "output",         "text": "..."}          # stdout (tool status + LLM tokens)
    {"type": "enforcement",    "mode": "strict"|"light"|"off"}  # active guidance-nudge level
    {"type": "mode",           "mode": "agent"|"plan"|"ask"}    # server-driven mode switch
                                                                # (plan approval → agent)
    {"type": "approval",       "id": "...", "tool": "...", "server": "...",
                               "args": {}, "risk": "...", "scope": "..."}
    {"type": "continue_prompt","id": "...", "summary": "..."}
    {"type": "user_question",  "id": "...", "questions": [
                               {"question": "...", "header": "...", "multiSelect": false,
                                "options": [{"label": "...", "description": "..."}]}]}
    {"type": "answer",         "text": "..."}          # final answer for a query
    {"type": "todo",           "items": [{"text": "...", "done": false}]}
    {"type": "error",          "text": "..."}
    {"type": "sessions_list",  "sessions": [{"id": "...", "title": "...",
                               "created_at": "...", "updated_at": "...", "preview": "..."}]}
    {"type": "session_loaded", "session_id": "...", "title": "...",
                               "display_messages": [...], "todos": [...]}
    {"type": "resources",      "resources": [{"uri": "...", "name": "...",
                               "description": "...", "mimeType": "..."}]}  # attachable resources

  Client → Server
    {"type": "query",             "text": "..."}   # @<uri> mentions are read & injected
    {"type": "list_resources"}                     # request the attachable-resource list
    {"type": "approval_response", "id": "...", "choice": "y"|"n"|"a"}
    {"type": "continue_response", "id": "...", "choice": "y"|"n"}
    {"type": "user_question_response", "id": "...", "answers": [
                               {"selected": ["..."], "otherText": "..."}]}
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
from .ws_session import _Session

import asyncio
import os
import socket
import sys
from typing import Any

try:
    import websockets
except ImportError as exc:
    raise ImportError(
        "websockets is required: pip install websockets"
    ) from exc


__all__ = ["serve", "main", "_AgentWorker", "_Session"]


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
    if not _model and os.environ.get("LLM_BACKEND", "").lower() == "vllm":
        # "Connect to running server" mode: no model was picked, so use whatever
        # the vLLM endpoint is already serving (first model from /v1/models).
        from ...query_engine.backends.vllm_backend import list_served_models
        served = list_served_models()
        if served:
            _model = served[0]
            print(f"Auto-selected served vLLM model: {_model}", file=_ORIGINAL_STDOUT)
    if not _model:
        raise ValueError(
            "No model specified. Pass --model <name> or set MIMIR_DEFAULT_MODEL."
        )

    # Prime the context-window cache (vLLM /v1/models or Ollama /api/show) so the
    # budget checks on the WS event loop hit the cache instead of blocking.
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

    print(f"MIMIR WS server starting on ws://{host}:{port}  (model: {_model})", file=_ORIGINAL_STDOUT)
    print("Initialising agent connections…", file=_ORIGINAL_STDOUT)

    worker = _AgentWorker(_model)
    print("Agent ready.", file=_ORIGINAL_STDOUT)

    async def _handler(ws: Any) -> None:
        session = _Session(ws, worker)
        await session.run()

    async with websockets.serve(_handler, host, port):
        print(f"Listening on ws://{host}:{port}", file=_ORIGINAL_STDOUT)
        # Print SLURM_NODE only after the socket is bound and ready to accept connections
        print(f"SLURM_NODE:{socket.gethostname()}", file=_ORIGINAL_STDOUT, flush=True)
        await asyncio.Future()  # run forever


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MIMIR WebSocket server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cwd", default=None, help="Set working directory before starting")
    parser.add_argument("--backend", choices=["ollama", "vllm", "anthropic"], default=None,
                        help="LLM backend to use (default: from LLM_BACKEND env var or vllm). "
                             "anthropic uses the hosted Claude API — the key comes from the "
                             "ANTHROPIC_API_KEY env var, never a CLI arg.")
    parser.add_argument("--vllm-base-url", default=None,
                        help="Base URL of the vLLM OpenAI-compatible API (overrides VLLM_BASE_URL env var)")
    parser.add_argument("--vllm-api-key", default=None,
                        help="API key for vLLM (default: EMPTY)")
    args = parser.parse_args()

    if args.cwd:
        os.chdir(args.cwd)

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    if args.vllm_api_key:
        os.environ["VLLM_API_KEY"] = args.vllm_api_key

    asyncio.run(serve(host=args.host, port=args.port, model=args.model))


if __name__ == "__main__":
    import pathlib as _pathlib
    _repo_root = str(_pathlib.Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    main()
