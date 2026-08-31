"""The append-only record of a session.

Neither list in the session file is a complete account: `display_messages` is what the
UI shows, and `llm_history` is a window the budget trims in place. The JSONL log is the
one thing nothing rewrites, which is what makes it usable both to rebuild the full
context and to profile a run offline — so these tests pin that it only ever grows, that
it stays readable, and that it can never take the drain loop down with it.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws import transcript_log
from mimir.client.ui.ws.transcript_log import TranscriptLog, read_transcript


class _TmpState:
    """Point the log at a scratch state dir for the duration of a test."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(transcript_log, "_MIMIR_DIR_WS", self._tmp.name)
        self._patch.start()
        return self._tmp.name

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()


class AppendTests(unittest.TestCase):
    def test_every_event_becomes_one_stamped_line(self):
        with _TmpState():
            log = TranscriptLog()
            log.bind("s1")
            log.append({"type": "query", "text": "hi"})
            log.append({"type": "tool_result", "id": "c1", "ok": True, "duration_ms": 42})
            events = read_transcript("s1")
        self.assertEqual([e["type"] for e in events], ["query", "tool_result"])
        self.assertEqual([e["seq"] for e in events], [1, 2])
        self.assertEqual(events[1]["duration_ms"], 42)  # payload kept for profiling
        self.assertTrue(events[0]["ts"])

    def test_streamed_deltas_are_left_out(self):
        """Hundreds per turn, and their content lands in the event that closes them."""
        with _TmpState():
            log = TranscriptLog()
            log.bind("s1")
            for _ in range(50):
                log.append({"type": "token", "text": "x"})
                log.append({"type": "thinking", "text": "y"})
            log.append({"type": "answer", "text": "done"})
            events = read_transcript("s1")
        self.assertEqual([e["type"] for e in events], ["answer"])
        self.assertEqual(events[0]["seq"], 1)

    def test_the_file_only_ever_grows(self):
        with _TmpState():
            log = TranscriptLog()
            log.bind("s1")
            log.append({"type": "query", "text": "first"})
            # A reconnect makes a fresh log object for the same session.
            again = TranscriptLog()
            again.bind("s1")
            again.append({"type": "query", "text": "second"})
            events = read_transcript("s1")
        self.assertEqual([e["text"] for e in events], ["first", "second"])
        self.assertEqual([e["seq"] for e in events], [1, 2])  # numbering resumed

    def test_no_session_means_nothing_is_written(self):
        with _TmpState():
            log = TranscriptLog()
            log.append({"type": "query", "text": "hi"})  # never bound
        # Nothing to assert but the absence of a crash and of a path to write to.
        self.assertIsNone(transcript_log.transcript_path(None))

    def test_a_write_failure_is_survivable(self):
        """It is teed off the drain loop, which must keep serving the client."""
        with _TmpState():
            log = TranscriptLog()
            log.bind("s1")
            with mock.patch("builtins.open", side_effect=OSError("read-only fs")):
                log.append({"type": "query", "text": "hi"})
            log.append({"type": "query", "text": "second"})
            events = read_transcript("s1")
        # The failed write left no hole in the numbering.
        self.assertEqual([e["seq"] for e in events], [1])
        self.assertEqual(events[0]["text"], "second")


class ReadTests(unittest.TestCase):
    def test_a_torn_line_does_not_lose_the_rest_of_the_file(self):
        with _TmpState() as state:
            path = transcript_log.transcript_path("s1")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"seq": 1, "type": "query"}) + "\n")
                f.write("{not json\n")
                f.write(json.dumps({"seq": 3, "type": "answer"}) + "\n")
            events = read_transcript("s1")
        self.assertEqual([e["seq"] for e in events], [1, 3])

    def test_an_unlogged_session_reads_as_empty(self):
        with _TmpState():
            self.assertEqual(read_transcript("never-used"), [])


if __name__ == "__main__":
    unittest.main()
