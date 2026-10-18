"""The WebSocket server's entry point (``main``), for the VS Code extension or any client.

Imports nothing of the client until ``--cwd`` is applied: ``config.paths`` resolves the
workspace at import. ``server/`` serves, ``connection/`` handles a socket, ``pool/``
holds a worker per conversation, ``worker/`` runs one agent. The wire protocol is in
CLIENT_DETAILED.md, "The wire protocol".
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def main() -> None:
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
        # Pin the sandbox/workspace root so the constants and the MCP servers all
        # resolve `.mimir` (and MCP_FILES_ROOT) to the same place.
        os.environ.setdefault("MCP_FILES_ROOT", os.getcwd())

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

    # Only now: everything below resolves the workspace root as it is imported.
    from .serving import serve
    asyncio.run(serve(host=args.host, port=args.port, model=args.model))


if __name__ == "__main__":
    import pathlib as _pathlib
    _repo_root = str(_pathlib.Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    main()
