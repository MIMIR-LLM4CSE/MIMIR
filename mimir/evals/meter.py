"""The meter: an OpenAI-compatible relay in front of the served models, recording every
completion request the same way whichever agent sent it — the run (its bearer key), the
model, the prompt, completion and prefix-cached tokens, the time to the first token and
to the end. A streamed request is asked for its usage (``stream_options``).

``python -m mimir.evals.meter serve --upstream URL --log meter.jsonl`` serves;
``python -m mimir.evals.meter summary meter.jsonl`` totals the log per run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

COMPLETION_PATHS = ("/chat/completions", "/completions")
HOP_HEADERS = {"host", "content-length", "authorization", "connection", "transfer-encoding",
               "accept-encoding"}
TIMEOUT_SECS = 3600.0


class Meter:
    """Where requests go (*upstreams* by model, *default* for the rest) and where each is
    recorded (*log_path*). *api_key* is what the upstream is called with."""

    def __init__(self, default: str | None, upstreams: dict[str, str], log_path: str,
                 api_key: str = "", force_usage: bool = True) -> None:
        if not default and not upstreams:
            raise ValueError("the meter needs an upstream")
        self.default = default.rstrip("/") if default else None
        self.upstreams = {m: u.rstrip("/") for m, u in upstreams.items()}
        self.log_path = log_path
        self.api_key = api_key
        self.force_usage = force_usage
        self._lock = threading.Lock()
        self.client = httpx.Client(timeout=TIMEOUT_SECS)

    def upstream_for(self, model: str | None) -> str:
        if model and model in self.upstreams:
            return self.upstreams[model]
        if self.default:
            return self.default
        raise KeyError(f"no upstream serves {model!r}")

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry) + "\n"
        with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line)


def _usage(block: Any) -> dict[str, int | None]:
    """The sizes an OpenAI ``usage`` block reports."""
    usage = block if isinstance(block, dict) else {}
    details = usage.get("prompt_tokens_details") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None}


class _Handler(BaseHTTPRequestHandler):
    meter: Meter
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — the base's name
        return

    def _run_tag(self) -> str:
        auth = self.headers.get("Authorization", "")
        return self.headers.get("X-Run-Tag") or (auth[7:] if auth.startswith("Bearer ") else "")

    def _forward_headers(self) -> dict[str, str]:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        if self.meter.api_key:
            headers["Authorization"] = f"Bearer {self.meter.api_key}"
        return headers

    def _path(self) -> str:
        # An agent given the meter's base without /v1 asks for the API's routes without
        # it; the server's own routes (/health, /tokenize) are at the root, as they are.
        bare = self.path.split("?", 1)[0].rstrip("/")
        return "/v1" + self.path if bare in ("/chat/completions", "/completions",
                                             "/models") else self.path

    def do_GET(self) -> None:  # noqa: N802 — http.server's name
        path = self._path()
        if path.rstrip("/") == "/v1/models" and self.meter.upstreams and not self.meter.default:
            body = json.dumps({"object": "list", "data": [
                {"id": m, "object": "model"} for m in self.meter.upstreams]}).encode()
            return self._send(200, {"Content-Type": "application/json"}, body)
        try:
            upstream = self.meter.upstream_for(None)
        except KeyError as exc:
            return self._send(404, {"Content-Type": "text/plain"}, str(exc).encode())
        response = self.meter.client.get(upstream + path, headers=self._forward_headers())
        self._send(response.status_code, {"Content-Type": response.headers.get(
            "content-type", "application/json")}, response.content)

    def do_POST(self) -> None:  # noqa: N802 — http.server's name
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = self._path()
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        model = body.get("model") if isinstance(body, dict) else None
        try:
            upstream = self.meter.upstream_for(model)
        except KeyError as exc:
            return self._send(404, {"Content-Type": "text/plain"}, str(exc).encode())
        metered = isinstance(body, dict) and path.endswith(COMPLETION_PATHS)
        stream = bool(metered and body.get("stream"))
        if stream and self.meter.force_usage:
            body.setdefault("stream_options", {})["include_usage"] = True
            raw = json.dumps(body).encode()
        entry: dict[str, Any] = {
            "ts": time.time(), "run_tag": self._run_tag(), "model": model, "path": path,
            "stream": stream,
            "messages": len(body.get("messages") or []) if metered else None,
            "tools": len(body.get("tools") or []) if metered else None,
        }
        started = time.monotonic()
        request = self.meter.client.build_request("POST", upstream + path, content=raw,
                                                  headers=self._forward_headers())
        try:
            response = self.meter.client.send(request, stream=True)
        except httpx.HTTPError as exc:
            if metered:
                self.meter.record({**entry, "status": 502, "error": type(exc).__name__,
                                   "latency_s": round(time.monotonic() - started, 3)})
            return self._send(502, {"Content-Type": "text/plain"},
                              f"the upstream did not answer: {exc}".encode())
        # Recorded before the answer is complete: an agent that exits on its last byte,
        # and whoever reads the log then, must find the call there.
        def finish(usage: Any) -> None:
            if metered:
                self.meter.record({**entry, "status": response.status_code, **_usage(usage),
                                   "latency_s": round(time.monotonic() - started, 3)})

        try:
            if stream:
                self._relay_stream(response, entry, started, finish)
            else:
                content = response.read()
                try:
                    usage = (json.loads(content) or {}).get("usage")
                except ValueError:
                    usage = None
                finish(usage)
                self._send(response.status_code, {"Content-Type": response.headers.get(
                    "content-type", "application/json")}, content)
        finally:
            response.close()

    def _relay_stream(self, response: httpx.Response, entry: dict[str, Any],
                      started: float, finish: Any) -> None:
        """Pass the event stream through as it comes, reading the usage and the first token;
        *finish* gets the usage before ``[DONE]`` (or the stream's end) goes."""
        self.send_response(response.status_code)
        self.send_header("Content-Type", response.headers.get("content-type", "text/event-stream"))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        usage, finished = None, False
        for line in response.iter_lines():
            data = line[5:].strip() if line.startswith("data:") else ""
            if data == "[DONE]" and not finished:  # a client may stop reading here
                finish(usage)
                finished = True
            if data and data != "[DONE]":
                try:
                    chunk = json.loads(data)
                except ValueError:
                    chunk = {}
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if "ttft_s" not in entry and any(
                        (c.get("delta") or {}).get(k)
                        for c in chunk.get("choices") or []
                        for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    entry["ttft_s"] = round(time.monotonic() - started, 3)
            self._chunk((line + "\n").encode())
        if not finished:
            finish(usage)
        self._chunk(b"")

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(meter: Meter, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """The meter's HTTP server, bound but not yet serving (``serve_forever``)."""
    handler = type("MeterHandler", (_Handler,), {"meter": meter})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def summarise(paths: list[str]) -> dict[str, dict[str, Any]]:
    """The log's requests totalled per run tag: calls, tokens, peak prompt, model time."""
    runs: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cached_tokens": 0, "peak_prompt_tokens": 0, "model_secs": 0.0, "models": set()})
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                run = runs[entry.get("run_tag") or ""]
                run["calls"] += 1
                run["errors"] += int(entry.get("status", 200) >= 400)
                for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                    run[key] += entry.get(key) or 0
                run["peak_prompt_tokens"] = max(run["peak_prompt_tokens"],
                                                entry.get("prompt_tokens") or 0)
                run["model_secs"] += entry.get("latency_s") or 0.0
                if entry.get("model"):
                    run["models"].add(entry["model"])
    return {tag: {**run, "models": sorted(run["models"]),
                  "model_secs": round(run["model_secs"], 3)} for tag, run in runs.items()}


def _upstreams(pairs: list[str]) -> tuple[str | None, dict[str, str]]:
    default, by_model = None, {}
    for pair in pairs:
        model, sep, url = pair.partition("=")
        if sep and not model.startswith(("http://", "https://")):
            by_model[model] = url
        else:
            default = pair
    return default, by_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mimir.evals.meter")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="relay and record the requests to the served models")
    serve.add_argument("--upstream", action="append", required=True, metavar="[MODEL=]URL",
                       help="the server of MODEL, or of every model not named (repeatable)")
    serve.add_argument("--listen", default="127.0.0.1:8099", metavar="HOST:PORT")
    serve.add_argument("--log", required=True, help="the JSON-lines file appended to")
    serve.add_argument("--api-key-env", default="VLLM_API_KEY",
                       help="the variable holding the key the upstream is called with")
    serve.add_argument("--no-force-usage", action="store_true",
                       help="leave a streamed request's stream_options as the agent sent them")
    summary = sub.add_parser("summary", help="total a log per run")
    summary.add_argument("logs", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "summary":
        json.dump(summarise(args.logs), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    default, by_model = _upstreams(args.upstream)
    meter = Meter(default, by_model, args.log, api_key=os.environ.get(args.api_key_env, ""),
                  force_usage=not args.no_force_usage)
    host, _, port = args.listen.rpartition(":")
    server = make_server(meter, host or "127.0.0.1", int(port))
    print(f"meter: http://{server.server_address[0]}:{server.server_address[1]}/v1 → "
          f"{default or ''} {by_model or ''}; log {args.log}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
