"""Hermetic tests for user-attached context via @-mention / attach.

Covers the frontend-agnostic core with no live servers:
- MCP-resource mention parsing (URIs, name shorthand, unknown tokens, punctuation)
- workspace-file mention parsing (whole file, line ranges, missing files) + slice reads
- attachment-block formatting
- ``agent.read_resource`` dispatch to the owning session
- ``register_resources`` registry population
- ``augment_query_with_resources`` end-to-end (parse → read → prepend), files + resources
"""

import asyncio
import os
import tempfile
import types
import unittest

from mimir.client.context.resource_context import (
    augment_query_with_resources,
    build_attachment_block,
    parse_file_mentions,
    parse_resource_mentions,
    read_file_slice,
)
from mimir.client.integration.server_manager import register_resources
from mimir.client.agent_core import MimirAgent


def _run(coro):
    # asyncio.run makes a fresh loop per call — robust when other tests in the
    # suite have closed/replaced the global loop.
    return asyncio.run(coro)


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _FakeContent:
    def __init__(self, text=None):
        self.text = text


class _FakeReadResult:
    def __init__(self, contents):
        self.contents = contents


class _FakeResource:
    def __init__(self, uri, name="", description="", mimeType=None):
        self.uri = uri
        self.name = name
        self.description = description
        self.mimeType = mimeType


class _FakeListResult:
    def __init__(self, resources):
        self.resources = resources


class _FakeSession:
    """Minimal MCP session double: canned list_resources / read_resource."""

    def __init__(self, resources=None, contents_by_uri=None, raise_list=False):
        self._resources = resources or []
        self._contents = contents_by_uri or {}
        self._raise_list = raise_list

    async def list_resources(self):
        if self._raise_list:
            raise RuntimeError("capability not supported")
        return _FakeListResult(self._resources)

    async def read_resource(self, uri):
        return _FakeReadResult(self._contents.get(str(uri), []))


_REGISTRY = {
    "memory://all": {"name": "memory", "description": "all stored memories"},
    "files://list": {"name": "files", "description": "workspace files"},
}


# ── Mention parsing ─────────────────────────────────────────────────────────────

class ParseMentionTests(unittest.TestCase):
    def test_matches_full_uri(self):
        cleaned, uris = parse_resource_mentions("summarize @memory://all please", _REGISTRY)
        self.assertEqual(uris, ["memory://all"])
        self.assertEqual(cleaned, "summarize please")

    def test_name_shorthand(self):
        _, uris = parse_resource_mentions("look at @files", _REGISTRY)
        self.assertEqual(uris, ["files://list"])

    def test_multiple_mentions_deduped_and_ordered(self):
        _, uris = parse_resource_mentions("@files and @memory://all and @files", _REGISTRY)
        self.assertEqual(uris, ["files://list", "memory://all"])

    def test_unknown_token_left_untouched(self):
        cleaned, uris = parse_resource_mentions("ping @nope://x now", _REGISTRY)
        self.assertEqual(uris, [])
        self.assertEqual(cleaned, "ping @nope://x now")

    def test_email_not_matched(self):
        # '@' preceded by non-whitespace is not a mention (negative lookbehind).
        cleaned, uris = parse_resource_mentions("mail me@x.com about @files", _REGISTRY)
        self.assertEqual(uris, ["files://list"])
        self.assertIn("me@x.com", cleaned)

    def test_trailing_punctuation_preserved(self):
        cleaned, uris = parse_resource_mentions("see @memory://all.", _REGISTRY)
        self.assertEqual(uris, ["memory://all"])
        self.assertIn(".", cleaned)

    def test_empty_registry_no_match(self):
        cleaned, uris = parse_resource_mentions("use @files", {})
        self.assertEqual(uris, [])
        self.assertEqual(cleaned, "use @files")


# ── Attachment block ────────────────────────────────────────────────────────────

class AttachmentBlockTests(unittest.TestCase):
    def test_formatting(self):
        block = build_attachment_block([("memory://all", "memory", "hello world")])
        self.assertIn("[Attached resource: memory://all (memory)]", block)
        self.assertIn("hello world", block)

    def test_empty_content_labelled(self):
        block = build_attachment_block([("files://list", "files", "   ")])
        self.assertIn("(empty)", block)

    def test_multiple_delimited(self):
        block = build_attachment_block([
            ("memory://all", "memory", "a"),
            ("files://list", "files", "b"),
        ])
        self.assertEqual(block.count("[Attached resource:"), 2)
        self.assertIn("\n---\n", block)


# ── read_resource dispatch ──────────────────────────────────────────────────────

class ReadResourceTests(unittest.TestCase):
    def _agent(self, sessions, resources):
        # Bind the real method onto a lightweight stub to avoid MimirAgent init.
        stub = types.SimpleNamespace(sessions=sessions, resources=resources)
        stub.read_resource = types.MethodType(MimirAgent.read_resource, stub)
        return stub

    def test_dispatch_joins_text(self):
        session = _FakeSession(contents_by_uri={
            "memory://all": [_FakeContent("line1"), _FakeContent("line2")],
        })
        agent = self._agent(
            {"memory": session},
            {"memory://all": {"name": "memory", "session": "memory"}},
        )
        self.assertEqual(_run(agent.read_resource("memory://all")), "line1\nline2")

    def test_unknown_uri_returns_empty(self):
        agent = self._agent({}, {})
        self.assertEqual(_run(agent.read_resource("nope://x")), "")

    def test_read_error_returns_empty(self):
        class _Boom(_FakeSession):
            async def read_resource(self, uri):
                raise RuntimeError("transport down")

        agent = self._agent(
            {"s": _Boom()},
            {"u://1": {"name": "u", "session": "s"}},
        )
        self.assertEqual(_run(agent.read_resource("u://1")), "")

    def test_blob_content_skipped(self):
        session = _FakeSession(contents_by_uri={
            "u://1": [_FakeContent(None), _FakeContent("kept")],
        })
        agent = self._agent(
            {"s": session},
            {"u://1": {"name": "u", "session": "s"}},
        )
        self.assertEqual(_run(agent.read_resource("u://1")), "kept")


# ── register_resources ──────────────────────────────────────────────────────────

class RegisterResourcesTests(unittest.TestCase):
    def test_populates_registry_with_owner(self):
        agent = types.SimpleNamespace(resources={})
        session = _FakeSession(resources=[
            _FakeResource("memory://all", name="memory", description="d", mimeType="text/plain"),
        ])
        n = _run(register_resources(agent=agent, name="mem_server", session=session))
        self.assertEqual(n, 1)
        self.assertEqual(agent.resources["memory://all"], {
            "name": "memory",
            "description": "d",
            "mimeType": "text/plain",
            "session": "mem_server",
        })

    def test_unsupported_capability_is_noop(self):
        agent = types.SimpleNamespace(resources={})
        session = _FakeSession(raise_list=True)
        n = _run(register_resources(agent=agent, name="s", session=session))
        self.assertEqual(n, 0)
        self.assertEqual(agent.resources, {})

    def test_name_defaults_to_uri(self):
        agent = types.SimpleNamespace(resources={})
        session = _FakeSession(resources=[_FakeResource("x://y", name="")])
        _run(register_resources(agent=agent, name="s", session=session))
        self.assertEqual(agent.resources["x://y"]["name"], "x://y")


# ── augment_query_with_resources (end-to-end) ───────────────────────────────────

class AugmentQueryTests(unittest.TestCase):
    def _agent(self):
        session = _FakeSession(contents_by_uri={
            "memory://all": [_FakeContent("remembered fact")],
        })
        stub = types.SimpleNamespace(
            sessions={"mem": session},
            resources={"memory://all": {"name": "memory", "session": "mem"}},
        )
        stub.read_resource = types.MethodType(MimirAgent.read_resource, stub)
        return stub

    def test_injects_content_and_reports_uri(self):
        effective, attached = _run(
            augment_query_with_resources(self._agent(), "summarize @memory://all")
        )
        self.assertEqual(attached, ["memory://all"])
        self.assertIn("remembered fact", effective)
        self.assertIn("summarize", effective)

    def test_no_mention_returns_text_unchanged(self):
        agent = self._agent()
        effective, attached = _run(augment_query_with_resources(agent, "no mentions here"))
        self.assertEqual((effective, attached), ("no mentions here", []))

    def test_no_at_sign_short_circuits(self):
        agent = self._agent()
        effective, attached = _run(augment_query_with_resources(agent, "plain text"))
        self.assertEqual((effective, attached), ("plain text", []))


# ── Workspace-file mentions ─────────────────────────────────────────────────────

class FileMentionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.rel = "pkg/mod.py"
        path = os.path.join(self.root, self.rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(f"line{i}\n" for i in range(1, 11)))  # line1..line10

    def tearDown(self):
        self._tmp.cleanup()

    def test_whole_file(self):
        cleaned, specs = parse_file_mentions(f"open @{self.rel} now", cwd=self.root)
        self.assertEqual(cleaned, "open now")
        self.assertEqual(len(specs), 1)
        self.assertEqual((specs[0].display, specs[0].start, specs[0].end), (self.rel, None, None))
        self.assertEqual(read_file_slice(specs[0]).count("\n"), 10)

    def test_line_range(self):
        _, specs = parse_file_mentions(f"see @{self.rel}:3-5", cwd=self.root)
        self.assertEqual((specs[0].start, specs[0].end), (3, 5))
        self.assertEqual(specs[0].display, f"{self.rel}:3-5")
        self.assertEqual(read_file_slice(specs[0]), "line3\nline4\nline5\n")

    def test_single_line(self):
        _, specs = parse_file_mentions(f"@{self.rel}:7", cwd=self.root)
        self.assertEqual((specs[0].start, specs[0].end), (7, 7))
        self.assertEqual(specs[0].display, f"{self.rel}:7")
        self.assertEqual(read_file_slice(specs[0]), "line7\n")

    def test_missing_file_left_untouched(self):
        cleaned, specs = parse_file_mentions("ref @pkg/nope.py here", cwd=self.root)
        self.assertEqual(specs, [])
        self.assertEqual(cleaned, "ref @pkg/nope.py here")

    def test_trailing_punctuation_preserved(self):
        cleaned, specs = parse_file_mentions(f"look at @{self.rel}.", cwd=self.root)
        self.assertEqual(len(specs), 1)
        self.assertTrue(cleaned.endswith("."))

    def test_dedup(self):
        _, specs = parse_file_mentions(f"@{self.rel} and @{self.rel}", cwd=self.root)
        self.assertEqual(len(specs), 1)

    def test_range_out_of_bounds_is_clamped(self):
        _, specs = parse_file_mentions(f"@{self.rel}:8-99", cwd=self.root)
        # Only existing lines 8..10 are returned; no error.
        self.assertEqual(read_file_slice(specs[0]), "line8\nline9\nline10\n")

    def test_unreadable_returns_note_not_raise(self):
        from mimir.client.context.resource_context import _FileSpec
        spec = _FileSpec(display="gone.py", abspath=os.path.join(self.root, "gone.py"),
                         start=None, end=None)
        self.assertTrue(read_file_slice(spec).startswith("(could not read file"))

    def test_whole_file_cap(self):
        big = os.path.join(self.root, "big.txt")
        with open(big, "w", encoding="utf-8") as fh:
            fh.write("x" * 50_000)
        _, specs = parse_file_mentions("@big.txt", cwd=self.root)
        out = read_file_slice(specs[0])
        self.assertIn("truncated", out)
        self.assertLess(len(out), 50_000)

    def test_resource_uri_not_treated_as_file(self):
        # A scheme:// token must not be mistaken for a file path.
        cleaned, specs = parse_file_mentions("use @memory://all now", cwd=self.root)
        self.assertEqual(specs, [])
        self.assertIn("@memory://all", cleaned)


class AugmentFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        with open(os.path.join(self.root, "a.py"), "w", encoding="utf-8") as fh:
            fh.write("alpha\nbeta\ngamma\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_file_attach_without_registry(self):
        agent = types.SimpleNamespace(resources={})
        eff, labels = asyncio.run(
            augment_query_with_resources(agent, "explain @a.py:1-2", cwd=self.root)
        )
        self.assertEqual(labels, ["a.py:1-2"])
        self.assertIn("[Attached file: a.py:1-2]", eff)
        self.assertIn("alpha", eff)
        self.assertIn("beta", eff)
        self.assertNotIn("gamma", eff)
        self.assertIn("explain", eff)

    def test_mixed_resource_and_file(self):
        session = _FakeSession(contents_by_uri={"memory://all": [_FakeContent("mem body")]})
        agent = types.SimpleNamespace(
            sessions={"mem": session},
            resources={"memory://all": {"name": "memory", "session": "mem"}},
        )
        agent.read_resource = types.MethodType(MimirAgent.read_resource, agent)
        eff, labels = asyncio.run(
            augment_query_with_resources(agent, "compare @memory and @a.py", cwd=self.root)
        )
        self.assertEqual(labels, ["memory://all", "a.py"])
        self.assertIn("mem body", eff)
        self.assertIn("alpha", eff)

    def test_no_mention_unchanged(self):
        agent = types.SimpleNamespace(resources={})
        eff, labels = asyncio.run(
            augment_query_with_resources(agent, "no attachments", cwd=self.root)
        )
        self.assertEqual((eff, labels), ("no attachments", []))


if __name__ == "__main__":
    unittest.main()
