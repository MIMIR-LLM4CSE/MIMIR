"""The neutral meter, against a stand-in server: what it relays, what it records, per run."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from mimir.evals.meter import Meter, make_server, summarise

USAGE = {"prompt_tokens": 120, "completion_tokens": 9,
         "prompt_tokens_details": {"cached_tokens": 100}}


class _Upstream(BaseHTTPRequestHandler):
    """A served model: a JSON answer, or an event stream ending on its usage when asked."""

    seen: list[dict] = []
    paths: list[str] = []

    def log_message(self, *args) -> None:
        return

    def do_GET(self) -> None:
        _Upstream.paths.append(self.path)
        body = json.dumps({"object": "list", "data": [{"id": "up"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        _Upstream.paths.append(self.path)
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append({"body": request, "auth": self.headers.get("Authorization")})
        if not request.get("stream"):
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                               "usage": USAGE}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [{"choices": [{"delta": {"content": "h"}}]},
                  {"choices": [{"delta": {"content": "i"}}]}]
        if (request.get("stream_options") or {}).get("include_usage"):
            events.append({"choices": [], "usage": USAGE})
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


def _serve(server: ThreadingHTTPServer) -> str:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


class MeterTests(unittest.TestCase):
    def setUp(self) -> None:
        _Upstream.seen = []
        _Upstream.paths = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        self.addCleanup(upstream.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = os.path.join(tmp.name, "meter.jsonl")
        meter = Meter(_serve(upstream), {}, self.log, api_key="upstream-key")
        server = make_server(meter)
        self.addCleanup(server.shutdown)
        self.base = _serve(server)

    def _post(self, tag: str, **body) -> httpx.Response:
        return httpx.post(f"{self.base}/v1/chat/completions",
                          headers={"Authorization": f"Bearer {tag}"},
                          json={"model": "glm", "messages": [{"role": "user", "content": "q"}],
                                **body}, timeout=10)

    def _entries(self) -> list[dict]:
        with open(self.log) as fh:
            return [json.loads(line) for line in fh]

    def test_a_plain_answer_is_relayed_and_its_usage_recorded_under_its_run(self) -> None:
        response = self._post("run-a")
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "hi")
        (entry,) = self._entries()
        self.assertEqual((entry["run_tag"], entry["model"], entry["stream"]), ("run-a", "glm", False))
        self.assertEqual((entry["prompt_tokens"], entry["completion_tokens"],
                          entry["cached_tokens"]), (120, 9, 100))
        self.assertEqual(entry["messages"], 1)

    def test_the_upstream_is_called_with_the_meter_s_key_not_the_run_s(self) -> None:
        self._post("run-a")
        self.assertEqual(_Upstream.seen[0]["auth"], "Bearer upstream-key")

    def test_a_stream_is_relayed_whole_and_asked_for_its_usage(self) -> None:
        with httpx.stream("POST", f"{self.base}/chat/completions",
                          headers={"Authorization": "Bearer run-b"},
                          json={"model": "glm", "messages": [], "stream": True},
                          timeout=10) as response:
            lines = [ln for ln in response.iter_lines() if ln.startswith("data:")]
        self.assertEqual(lines[-1], "data: [DONE]")
        self.assertEqual(len(lines), 4)
        self.assertTrue(_Upstream.seen[0]["body"]["stream_options"]["include_usage"])
        (entry,) = self._entries()
        self.assertTrue(entry["stream"])
        self.assertEqual(entry["prompt_tokens"], 120)
        self.assertIn("ttft_s", entry)

    def test_other_requests_pass_through_unrecorded(self) -> None:
        response = httpx.get(f"{self.base}/v1/models", timeout=10)
        self.assertEqual(response.json()["data"][0]["id"], "up")
        self.assertFalse(os.path.exists(self.log))

    def test_the_server_s_own_routes_stay_at_the_root(self) -> None:
        httpx.get(f"{self.base}/models", timeout=10)
        httpx.post(f"{self.base}/tokenize", json={"model": "glm", "prompt": "x"}, timeout=10)
        self.assertEqual(_Upstream.paths, ["/v1/models", "/tokenize"])

    def test_the_log_totals_per_run(self) -> None:
        self._post("run-a")
        self._post("run-a")
        self._post("run-b")
        totals = summarise([self.log])
        self.assertEqual(totals["run-a"]["calls"], 2)
        self.assertEqual(totals["run-a"]["prompt_tokens"], 240)
        self.assertEqual(totals["run-a"]["peak_prompt_tokens"], 120)
        self.assertEqual(totals["run-b"]["models"], ["glm"])


class DownstreamFailureTests(unittest.TestCase):
    def test_an_upstream_that_does_not_answer_is_a_502_on_record(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = os.path.join(tmp.name, "meter.jsonl")
        server = make_server(Meter("http://127.0.0.1:9", {}, log))
        self.addCleanup(server.shutdown)
        response = httpx.post(f"{_serve(server)}/v1/chat/completions",
                              json={"model": "m", "messages": []}, timeout=10)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(summarise([log])[""]["errors"], 1)


class RoutingTests(unittest.TestCase):
    def test_a_model_goes_to_its_own_server_and_an_unknown_one_is_refused(self) -> None:
        meter = Meter(None, {"glm": "http://glm:8000/", "small": "http://small:8000"}, "x")
        self.assertEqual(meter.upstream_for("glm"), "http://glm:8000")
        with self.assertRaises(KeyError):
            meter.upstream_for("other")

    def test_a_meter_with_nowhere_to_send_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Meter(None, {}, "x")


if __name__ == "__main__":
    unittest.main()
