"""Hermetic tests for the one-sentence session descriptions.

``session_summary`` builds a compact transcript, asks the backend for a single
descriptive sentence, and falls back to the first user message whenever the
model is unreachable or answers with nothing usable. No network here — the
backend is stubbed.
"""

import json
import unittest
from unittest.mock import patch

from mimir.client.ui.ws import session_summary as ss
from mimir.client.ui.ws.session_store import FullSession


def _msgs(n: int) -> list[dict]:
    out = []
    for i in range(n):
        out.append({"role": "user", "kind": "text", "text": f"q{i}"})
        out.append({"role": "agent", "kind": "text", "text": f"a{i}"})
    return out


class _StubBackend:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, model, messages, tools, thinking, streaming, options, **kw):
        self.calls.append((model, messages, tools, thinking, streaming, options))
        if isinstance(self.content, Exception):
            raise self.content
        return {"role": "assistant", "content": self.content}


class TranscriptTests(unittest.TestCase):
    def test_only_text_user_and_agent_messages_are_kept(self):
        text = ss._transcript([
            {"role": "user", "kind": "text", "text": "hello"},
            {"role": "system", "kind": "text", "text": "🔔 wake"},
            {"role": "agent", "kind": "diff", "text": "ignored"},
            {"role": "agent", "kind": "text", "text": "world"},
            {"role": "user", "kind": "text", "text": "   "},
        ])
        self.assertEqual(text, "User: hello\nAssistant: world")

    def test_long_messages_are_truncated(self):
        long = "x" * (ss._MSG_CHARS + 500)
        text = ss._transcript([{"role": "user", "kind": "text", "text": long}])
        self.assertEqual(len(text), len("User: ") + ss._MSG_CHARS)

    def test_head_and_tail_are_kept_for_long_sessions(self):
        lines = ss._transcript(_msgs(20)).split("\n")
        self.assertEqual(len(lines), ss._MAX_MSGS)
        self.assertEqual(lines[:2], ["User: q0", "Assistant: a0"])
        self.assertEqual(lines[-1], "Assistant: a19")


class CleanTests(unittest.TestCase):
    def test_strips_quotes_and_markdown_markers(self):
        self.assertEqual(ss._clean('  "- Fixed the parser."'), "Fixed the parser.")

    def test_keeps_the_last_line_when_the_model_adds_a_lead_in(self):
        self.assertEqual(
            ss._clean("Sure, here it is:\nRefactored the store."), "Refactored the store."
        )

    def test_strips_label_prefixes(self):
        self.assertEqual(ss._clean("Description: Fixed the parser."), "Fixed the parser.")

    def test_caps_length_on_a_word_boundary(self):
        out = ss._clean(" ".join(["word"] * 80))
        self.assertLessEqual(len(out), 161)
        self.assertTrue(out.endswith("…"))
        self.assertNotIn("wor…", out)  # never clipped mid-word


class GenerateTests(unittest.TestCase):
    def test_uses_backend_reply(self):
        stub = _StubBackend("Refactored the session store.")
        with patch.object(ss, "get_backend", return_value=stub):
            out, generated = ss.generate_summary("m", _msgs(1))
        self.assertEqual(out, "Refactored the session store.")
        self.assertTrue(generated)
        _, messages, tools, thinking, streaming, _ = stub.calls[0]
        self.assertEqual(tools, [])          # summarizing is not a tool task
        self.assertFalse(thinking)
        self.assertFalse(streaming)
        self.assertEqual(messages[0]["role"], "system")

    def test_generous_token_budget(self):
        # A tight budget lets a reasoning block swallow the whole completion.
        stub = _StubBackend("Did a thing.")
        with patch.object(ss, "get_backend", return_value=stub):
            ss.generate_summary("m", _msgs(1))
        options = stub.calls[0][5]
        self.assertEqual(options["max_tokens"], ss._MAX_TOKENS)
        self.assertEqual(options["num_predict"], ss._MAX_TOKENS)
        self.assertGreaterEqual(ss._MAX_TOKENS, 200)

    def test_backend_failure_falls_back_to_the_user_query(self):
        stub = _StubBackend(RuntimeError("backend down"))
        with patch.object(ss, "get_backend", return_value=stub):
            out, generated = ss.generate_summary("m", [
                {"role": "user", "kind": "text", "text": "Fix the\nparser bug"},
            ])
        self.assertEqual(out, "Fix the parser bug")
        self.assertFalse(generated)  # provisional: the next query retries

    def test_empty_reply_falls_back_to_the_user_query(self):
        stub = _StubBackend("   ")
        with patch.object(ss, "get_backend", return_value=stub):
            out, generated = ss.generate_summary(
                "m", [{"role": "user", "kind": "text", "text": "hi"}]
            )
        self.assertEqual(out, "hi")
        self.assertFalse(generated)

    def test_answer_inside_the_reasoning_block_is_salvaged(self):
        class _Thinking(_StubBackend):
            def chat(self, *a, **kw):
                super().chat(*a, **kw)
                return {"role": "assistant", "content": "",
                        "thinking": "Let me see.\nAdded the sessions panel."}

        with patch.object(ss, "get_backend", return_value=_Thinking("")):
            out, generated = ss.generate_summary("m", _msgs(1))
        self.assertEqual(out, "Added the sessions panel.")
        self.assertTrue(generated)

    def test_no_usable_transcript_skips_the_model(self):
        stub = _StubBackend("never called")
        with patch.object(ss, "get_backend", return_value=stub):
            out, generated = ss.generate_summary(
                "m", [{"role": "agent", "kind": "diff", "text": "d"}]
            )
        self.assertEqual((out, generated), ("", False))
        self.assertEqual(stub.calls, [])


class StoreRoundTripTests(unittest.TestCase):
    def test_summary_fields_survive_serialisation(self):
        s = FullSession(id="i", title="t", created_at="c", updated_at="u")
        s.summary = "Did a thing."
        s.summary_msgs = 4
        s.summary_version = ss.SUMMARY_VERSION
        s.title_custom = True
        back = FullSession.from_dict(s.to_dict())
        self.assertEqual(back.summary, "Did a thing.")
        self.assertEqual(back.summary_msgs, 4)
        self.assertEqual(back.summary_version, ss.SUMMARY_VERSION)
        self.assertTrue(back.title_custom)
        meta = back.meta()
        self.assertEqual(meta.summary, "Did a thing.")
        self.assertEqual(meta.summary_version, ss.SUMMARY_VERSION)
        self.assertTrue(meta.title_custom)

    def test_legacy_sessions_default_the_new_fields(self):
        back = FullSession.from_dict({"id": "i", "title": "old"})
        self.assertEqual(back.summary, "")
        self.assertEqual(back.summary_msgs, 0)
        self.assertEqual(back.summary_version, 0)
        self.assertFalse(back.title_custom)
        # Version 0 is older than the current generator, so such a summary is
        # hidden by the WS layer until it is regenerated.
        self.assertNotEqual(0, ss.SUMMARY_VERSION)


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))


class _FakeStore:
    def __init__(self, sessions):
        self.sessions = {s.id: s for s in sessions}
        self.saved = []

    def list_sessions(self):
        return [s.meta() for s in self.sessions.values()]

    def session_exists(self, sid):
        return sid in self.sessions

    def load_session(self, sid):
        return FullSession.from_dict(self.sessions[sid].to_dict())

    def save_session(self, session):
        self.sessions[session.id] = session
        self.saved.append(session)


def _session_with(**kw) -> FullSession:
    s = FullSession(id=kw.pop("id", "s1"), title=kw.pop("title", "first query"),
                    created_at="c", updated_at="u")
    s.display_messages = kw.pop("display_messages", _msgs(1))
    for key, val in kw.items():
        setattr(s, key, val)
    return s


class WSLayerTests(unittest.IsolatedAsyncioTestCase):
    """The WS glue: stale summaries are hidden, fresh ones are pushed."""

    def _session(self, store):
        from mimir.client.ui.ws.ws_session import _Session
        sess = object.__new__(_Session)
        sess.ws = _FakeWS()
        sess.store = store
        sess._unsaved_session_meta = None
        sess._active_session_id = "s1"
        sess._summary_task = None
        sess.worker = type("W", (), {"model": "m"})()
        return sess

    async def test_provisional_and_generated_summaries_are_both_sent(self):
        store = _FakeStore([
            _session_with(id="s1", summary="the raw query",
                          summary_version=ss.PROVISIONAL_VERSION),
            _session_with(id="s2", summary="Built the panel.",
                          summary_version=ss.SUMMARY_VERSION),
        ])
        sess = self._session(store)
        await sess._send_sessions_list()
        rows = {r["id"]: r["summary"] for r in sess.ws.sent[0]["sessions"]}
        self.assertEqual(rows["s1"], "the raw query")   # shown while it retries
        self.assertEqual(rows["s2"], "Built the panel.")

    async def test_refresh_stores_the_summary_and_pushes_the_list(self):
        store = _FakeStore([_session_with(id="s1")])
        sess = self._session(store)
        with patch.object(ss, "get_backend", return_value=_StubBackend("Built the panel.")):
            await sess._refresh_session_summary("s1")
        saved = store.sessions["s1"]
        self.assertEqual(saved.summary, "Built the panel.")
        self.assertEqual(saved.summary_msgs, 2)
        self.assertEqual(saved.summary_version, ss.SUMMARY_VERSION)
        self.assertEqual(sess.ws.sent[-1]["type"], "sessions_list")

    async def test_failed_generation_stores_the_query_as_provisional(self):
        store = _FakeStore([_session_with(
            id="s1", display_messages=[{"role": "user", "kind": "text", "text": "fix the parser"}],
        )])
        sess = self._session(store)
        with patch.object(ss, "get_backend", return_value=_StubBackend(RuntimeError("down"))):
            await sess._refresh_session_summary("s1")
        saved = store.sessions["s1"]
        self.assertEqual(saved.summary, "fix the parser")
        self.assertEqual(saved.summary_version, ss.PROVISIONAL_VERSION)

        # …and the next query retries the real generation rather than keeping it.
        saved.display_messages.append({"role": "agent", "kind": "text", "text": "done"})
        with patch.object(ss, "get_backend", return_value=_StubBackend("Fixed the parser.")):
            await sess._refresh_session_summary("s1")
        self.assertEqual(store.sessions["s1"].summary, "Fixed the parser.")
        self.assertEqual(store.sessions["s1"].summary_version, ss.SUMMARY_VERSION)

    async def test_fresh_summary_is_not_regenerated(self):
        store = _FakeStore([_session_with(
            id="s1", summary="Built the panel.", summary_msgs=2,
            summary_version=ss.SUMMARY_VERSION,
        )])
        sess = self._session(store)
        stub = _StubBackend("should not be called")
        with patch.object(ss, "get_backend", return_value=stub):
            await sess._refresh_session_summary("s1")
        self.assertEqual(stub.calls, [])
        self.assertEqual(store.saved, [])

    async def test_each_answered_turn_refines_the_description(self):
        # Summary covers 2 messages; a second answered turn makes it 4 → regenerate.
        store = _FakeStore([_session_with(
            id="s1", summary="Built the panel.", summary_msgs=2,
            summary_version=ss.SUMMARY_VERSION, display_messages=_msgs(2),
        )])
        sess = self._session(store)
        with patch.object(ss, "get_backend", return_value=_StubBackend("Built the panel, then fixed X.")):
            await sess._refresh_session_summary("s1")
        self.assertEqual(store.sessions["s1"].summary, "Built the panel, then fixed X.")
        self.assertEqual(store.sessions["s1"].summary_msgs, 4)

    async def test_the_answer_is_part_of_the_described_transcript(self):
        store = _FakeStore([_session_with(id="s1", display_messages=[
            {"role": "user", "kind": "text", "text": "add a mode switcher"},
            {"role": "agent", "kind": "text", "text": "Added ModeSwitcher.tsx and wired it up."},
        ])])
        sess = self._session(store)
        stub = _StubBackend("Added a mode switcher to the webview.")
        with patch.object(ss, "get_backend", return_value=stub):
            await sess._refresh_session_summary("s1")
        prompt = stub.calls[0][1][-1]["content"]
        self.assertIn("Assistant: Added ModeSwitcher.tsx", prompt)
        self.assertEqual(store.sessions["s1"].summary, "Added a mode switcher to the webview.")

    async def test_stale_version_forces_a_regeneration(self):
        store = _FakeStore([_session_with(
            id="s1", summary="old-style text", summary_msgs=2, summary_version=0,
        )])
        sess = self._session(store)
        with patch.object(ss, "get_backend", return_value=_StubBackend("Built the panel.")):
            await sess._refresh_session_summary("s1")
        self.assertEqual(store.sessions["s1"].summary, "Built the panel.")


if __name__ == "__main__":
    unittest.main()
