"""Tests for the code-intelligence server and the minimal LSP client.

Backend-selection ladder (LSP -> ctags -> scan) and graceful degradation are
exercised with the higher tiers monkeypatched, so the suite is hermetic and does
not require ctags or a language server to be installed.
"""
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import types
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SHARED = _ROOT / "servers" / "_shared"
import sys
sys.path.insert(0, str(_SHARED))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(_ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lsp_client = _load("lsp_client_t", "servers/_shared/lsp_client.py")
code_intel = _load("server_code_intel_t", "servers/workspace/server_code_intel.py")


def _frame(obj: dict) -> bytes:
    body = json.dumps(obj).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


class LSPClientWireTest(unittest.TestCase):
    def _client_with_stdout(self, messages: list[dict]) -> lsp_client.LSPClient:
        client = lsp_client.LSPClient(["dummy"], ".")
        stdout = io.BytesIO(b"".join(_frame(m) for m in messages))
        stdin = io.BytesIO()
        client._proc = types.SimpleNamespace(stdin=stdin, stdout=stdout)
        client._ready = True
        return client

    def test_request_parses_framed_response(self):
        client = self._client_with_stdout([{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}])
        out = client._request("workspace/symbol", {"query": "x"})
        self.assertEqual(out, {"ok": True})

    def test_request_drains_diagnostics_notification_before_response(self):
        client = self._client_with_stdout([
            {"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
             "params": {"uri": "file:///a.py", "diagnostics": [{"message": "boom"}]}},
            {"jsonrpc": "2.0", "id": 1, "result": []},
        ])
        out = client._request("workspace/symbol", {"query": "x"})
        self.assertEqual(out, [])
        self.assertEqual(client._diagnostics["file:///a.py"], [{"message": "boom"}])

    def test_missing_binary_start_is_false(self):
        client = lsp_client.LSPClient(["definitely-not-a-real-langserver-xyz"], ".")
        self.assertFalse(client.start())

    def test_extension_routing(self):
        self.assertIn(lsp_client.language_id("a.cu"), {"cuda"})
        self.assertEqual(lsp_client.language_id("a.f90"), "fortran")
        self.assertEqual(lsp_client.language_id("a.unknown"), "plaintext")


class CodeIntelBackendLadderTest(unittest.TestCase):
    def setUp(self):
        # Force the LSP tier off so we deterministically exercise ctags/scan.
        self._orig_client_for = code_intel._lsp.client_for
        code_intel._lsp.client_for = lambda path: None
        self._orig_index = code_intel._ctags_index
        self._orig_root = code_intel.SEARCH_ROOT

    def tearDown(self):
        code_intel._lsp.client_for = self._orig_client_for
        code_intel._ctags_index = self._orig_index
        code_intel.SEARCH_ROOT = self._orig_root

    def test_find_definition_uses_ctags_when_available(self):
        code_intel._ctags_index = lambda force=False: {
            # A real index stores absolute paths (see _entry_from -> _out_path).
            "defs": {"my_func": [{"path": "/repo/a.py", "line": 3, "kind": "function",
                                   "scope": "", "signature": "(x)"}]},
            "by_file": {},
        }
        res = code_intel.find_definition("my_func")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["backend"], "ctags")
        self.assertEqual(res["definitions"][0]["path"], "/repo/a.py")
        self.assertEqual(res["definitions"][0]["line"], 3)

    def test_find_definition_errs_when_no_backend(self):
        code_intel._ctags_index = lambda force=False: None
        res = code_intel.find_definition("whatever")
        self.assertEqual(res["status"], "error")
        self.assertIn("ctags", res.get("hint", "").lower() + res["error"].lower())

    def test_symbol_outline_uses_ctags(self):
        with tempfile.TemporaryDirectory() as d:
            code_intel.SEARCH_ROOT = d
            p = os.path.join(d, "a.py")
            with open(p, "w") as fh:
                fh.write("class B:\n    def m(self): ...\n")
            # by_file is keyed by the absolute path the outline looks up with.
            code_intel._ctags_index = lambda force=False: {
                "defs": {},
                "by_file": {os.path.abspath(p): [
                    {"name": "B", "kind": "class", "line": 1, "scope": "", "signature": ""},
                    {"name": "m", "kind": "member", "line": 2, "scope": "B",
                     "signature": "(self)"},
                ]},
            }
            res = code_intel.symbol_outline("a.py")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["backend"], "ctags")
        self.assertEqual([s["name"] for s in res["symbols"]], ["B", "m"])

    def test_find_references_scan_tier(self):
        with tempfile.TemporaryDirectory() as d:
            code_intel.SEARCH_ROOT = d
            code_intel._ctags_index = lambda force=False: None
            with open(os.path.join(d, "a.py"), "w") as fh:
                fh.write("def target():\n    return target\n# not_target here\n")
            res = code_intel.find_references("target")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["backend"], "scan")
        lines = sorted(r["line"] for r in res["references"])
        self.assertEqual(lines, [1, 2])  # whole-word: 'not_target' excluded

    def test_parse_traditional_and_json_tags_equivalent(self):
        """Both ctags output formats (Universal JSON, Exuberant extended) parse."""
        code_intel.SEARCH_ROOT = "/repo"
        trad = (
            "!_TAG_FILE_FORMAT\t2\t/extended format/\n"
            "build_x\tsrc/a.py\t/^def build_x(y):$/;\"\tfunction\tline:5\tsignature:(y)\n"
            "B\tsrc/a.py\t/^class B:$/;\"\tclass\tline:1\n"
            "m\tsrc/a.py\t/^    def m(self):$/;\"\tmember\tline:2\tclass:B\n"
        )
        idx_t = code_intel._parse_traditional_tags(trad)
        self.assertIn("build_x", idx_t["defs"])
        e = idx_t["defs"]["build_x"][0]
        self.assertEqual((e["path"], e["line"], e["kind"], e["signature"]),
                         ("/repo/src/a.py", 5, "function", "(y)"))
        self.assertEqual(idx_t["defs"]["m"][0]["scope"], "B")

        jline = json.dumps({"_type": "tag", "name": "build_x", "path": "src/a.py",
                            "line": 5, "kind": "function", "signature": "(y)"})
        idx_j = code_intel._parse_json_tags(jline + "\n")
        self.assertEqual(idx_j["defs"]["build_x"][0]["line"], 5)
        self.assertEqual(idx_j["defs"]["build_x"][0]["kind"], "function")

    def test_hover_requires_lsp(self):
        with tempfile.TemporaryDirectory() as d:
            code_intel.SEARCH_ROOT = d
            p = os.path.join(d, "a.py")
            with open(p, "w") as fh:
                fh.write("x = 1\n")
            res = code_intel.hover("a.py", 1, "x")
        self.assertEqual(res["status"], "error")
        self.assertIn("language server", res["error"].lower())


if __name__ == "__main__":
    unittest.main()
