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
            res = code_intel.symbol_outline(p)
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
            res = code_intel.hover(p, 1, "x")
        self.assertEqual(res["status"], "error")
        self.assertIn("language server", res["error"].lower())


class SearchCarriesSourceContextTest(unittest.TestCase):
    """A hit must be actionable on its own.

    ``{path, line}`` locates a symbol but cannot be edited against: an anchor needs the
    text exactly as it stands. Returning the line number alone is what makes a search
    cost a follow-up read.
    """

    def setUp(self):
        self._orig_client_for = code_intel._lsp.client_for
        code_intel._lsp.client_for = lambda path: None
        self._orig_index = code_intel._ctags_index
        self._orig_root = code_intel.SEARCH_ROOT
        self._tmp = tempfile.TemporaryDirectory()
        code_intel.SEARCH_ROOT = self._tmp.name
        code_intel._ctags_index = lambda force=False: None

    def tearDown(self):
        code_intel._lsp.client_for = self._orig_client_for
        code_intel._ctags_index = self._orig_index
        code_intel.SEARCH_ROOT = self._orig_root
        self._tmp.cleanup()

    def _write(self, body: str, name: str = "a.py") -> str:
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as fh:
            fh.write(body)
        return p

    def test_reference_carries_numbered_surrounding_source(self):
        self._write("".join(f"pad_{i} = {i}\n" for i in range(1, 10))
                    + "    value = target\n")
        res = code_intel.find_references("target", context_lines=3)
        ctx = res["references"][0]["context"]
        self.assertEqual(ctx["start_line"], 7)
        self.assertEqual(ctx["end_line"], 10)
        self.assertIn("10:     value = target", ctx["text"])

    def test_context_preserves_exact_indentation(self):
        self._write("def f():\n        deep = target\n")
        ctx = code_intel.find_references(
            "target", context_lines=3)["references"][0]["context"]
        self.assertIn("2:         deep = target", ctx["text"])

    def test_references_carry_no_excerpt_by_default(self):
        # "Where is it used" is answered by the matching line; an excerpt per hit
        # costs more than the answer.
        self._write("v = target\n")
        hit = code_intel.find_references("target")["references"][0]
        self.assertNotIn("context", hit)
        self.assertEqual(hit["text"], "v = target")

    def test_context_lines_zero_disables_it(self):
        self._write("v = target\n")
        res = code_intel.find_references("target", context_lines=0)
        self.assertNotIn("context", res["references"][0])

    def test_context_lines_is_clamped(self):
        self._write("".join(f"pad_{i} = {i}\n" for i in range(1, 200))
                    + "v = target\n")
        ctx = code_intel.find_references("target", context_lines=999)["references"][0]["context"]
        self.assertEqual(ctx["end_line"] - ctx["start_line"], code_intel._MAX_CONTEXT_LINES)

    def test_definition_carries_context_too(self):
        p = self._write("class B:\n    pass\n")
        code_intel._ctags_index = lambda force=False: {
            "defs": {"B": [{"path": p, "line": 1, "kind": "class",
                            "scope": "", "signature": ""}]},
            "by_file": {},
        }
        res = code_intel.find_definition("B")
        self.assertIn("1: class B:", res["definitions"][0]["context"]["text"])


class SymbolOutlineSpansTest(unittest.TestCase):
    """An outline says where each symbol ENDS, not only where it starts.

    Root cause this covers: the edit tools are addressed by line, so a map of start
    lines alone leaves "read the block around line N" to be found by widening the
    window five lines at a time — the crawl seen in the wild (1486-1520, 1515-1525,
    1510-1530, 1515-1535) before a single insert.
    """

    def test_flat_symbol_information_is_read_like_a_nested_one(self):
        # Two shapes answer documentSymbol. Reading only the nested one put every
        # symbol of a flat server at line 1 — an outline actively pointing the wrong way.
        out: list[dict] = []
        code_intel._flatten_doc_symbols([{
            "name": "solve", "kind": 12,
            "location": {"uri": "file:///f.py",
                         "range": {"start": {"line": 41}, "end": {"line": 58}}},
        }], out)
        self.assertEqual((out[0]["line"], out[0]["end_line"]), (42, 59))

    def test_end_comes_from_the_full_range_not_the_name(self):
        # selectionRange covers the identifier alone; using its end would report
        # every symbol as one line long.
        out: list[dict] = []
        code_intel._flatten_doc_symbols([{
            "name": "solve", "kind": 12,
            "range": {"start": {"line": 41}, "end": {"line": 58}},
            "selectionRange": {"start": {"line": 41}, "end": {"line": 41}},
        }], out)
        self.assertEqual((out[0]["line"], out[0]["end_line"]), (42, 59))

    def test_a_missing_end_is_derived_from_the_next_symbol(self):
        # Exuberant ctags reports no end field, so most clusters have no exact answer.
        entries = [{"name": "a", "line": 10, "end_line": 0, "depth": 0},
                   {"name": "b", "line": 30, "end_line": 0, "depth": 0}]
        filled = code_intel._fill_end_lines(entries, total_lines=100)
        self.assertEqual([(e["line"], e["end_line"]) for e in filled], [(10, 29), (30, 100)])
        self.assertTrue(all(e["inferred_end"] for e in filled))

    def test_a_nested_symbol_does_not_end_its_parent(self):
        entries = [{"name": "C", "line": 10, "end_line": 0, "depth": 0},
                   {"name": "C.m", "line": 12, "end_line": 0, "depth": 1},
                   {"name": "after", "line": 40, "end_line": 0, "depth": 0}]
        filled = code_intel._fill_end_lines(entries, total_lines=80)
        by_name = {e["name"]: e["end_line"] for e in filled}
        self.assertEqual(by_name["C"], 39)      # runs to the next top-level symbol
        self.assertEqual(by_name["C.m"], 39)    # its method, same upper bound
        self.assertEqual(by_name["after"], 80)

    def test_an_exact_end_is_left_alone(self):
        entries = [{"name": "a", "line": 10, "end_line": 20, "depth": 0},
                   {"name": "b", "line": 30, "end_line": 0, "depth": 0}]
        filled = code_intel._fill_end_lines(entries, total_lines=100)
        exact = next(e for e in filled if e["name"] == "a")
        self.assertEqual(exact["end_line"], 20)
        self.assertNotIn("inferred_end", exact)

    def test_derived_spans_never_overlap(self):
        # The -1 is what makes the spans a partition: the next symbol's first line is
        # the next symbol's, so without it every pair would share a line and each span
        # would carry the following declaration's header.
        entries = [{"name": f"s{i}", "line": 1 + i * 20, "end_line": 0, "depth": 0}
                   for i in range(5)]
        filled = code_intel._fill_end_lines(entries, total_lines=200)
        for earlier, later in zip(filled, filled[1:]):
            self.assertEqual(earlier["end_line"] + 1, later["line"])

    def test_the_rule_reads_only_line_and_depth(self):
        # Nothing here parses source, so a Fortran subroutine, a C function and a Java
        # method are the same problem. Kinds are passed through untouched.
        entries = [{"name": "sub_a", "kind": "subroutine", "line": 1439, "end_line": 0, "depth": 0},
                   {"name": "sub_b", "kind": "subroutine", "line": 1540, "end_line": 0, "depth": 0}]
        filled = code_intel._fill_end_lines(entries, total_lines=1786)
        self.assertEqual([(e["line"], e["end_line"]) for e in filled],
                         [(1439, 1539), (1540, 1786)])


if __name__ == "__main__":
    unittest.main()
