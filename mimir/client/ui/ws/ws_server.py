"""WebSocket server for MimirAgent.

Provides a streaming JSON protocol consumed by the VS Code extension (or any
WebSocket client).  The MimirAgent runs in a dedicated background thread with its
own asyncio event loop so blocking approval prompts never freeze the WebSocket
event loop.

This module is the entry point (``main``). It imports nothing of the client until
the command line is applied: ``--cwd`` has to be in force before ``config.paths`` and
the backend factory resolve the workspace root, which they do at import time. The
implementation is split across sibling modules:
  - ``serving``     — ``serve``: model resolution, bind, the shutdown path.
  - ``_ws_runtime`` — shared foundation (stdout router, todo helpers, context-budget
    constants).
  - ``ws_worker``   — ``_AgentWorker``, the background agent thread.
  - ``ws_pool``     — ``_AgentPool``, one worker per conversation: built on that
                      conversation's first query, released when it goes idle.
  - ``ws_session``  — ``_Session``, one WebSocket connection.

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
    {"type": "transcript_resync", "session_id": "..."}  # send the transcript whole
    {"type": "detached",       "detached": true, "log": "...",
                               "autonomy": "manual|auto|auto_all",
                               "sessions": [...], "pid": 0, "setsid": true}
                                                       # the answer to ``detach``, and
                                                       # sent again on attach to a server
                                                       # already claimed: it has
                                                       # re-pointed its output at ``log``
                                                       # and left the extension host's
                                                       # process group, so it survives
                                                       # the window. ``sessions`` are the
                                                       # conversations claiming it, and
                                                       # ``detached`` is whether any
                                                       # does — false means the next
                                                       # departure of the last client
                                                       # ends it
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
    {"type": "transcript",        "session_id": "...", "messages": [...],
                                  "epoch": "...", "rev": 0}
                                  # the client's rendered chat, stored verbatim as the
                                  # session's display messages (see _handle_transcript)
    {"type": "transcript",        "session_id": "...", "delta": true, "order": [ids],
                                  "changed": [...], "epoch": "...", "base_rev": n,
                                  "rev": n+1}
                                  # only what is new or replaced since base_rev; a
                                  # server not holding it answers transcript_resync
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
