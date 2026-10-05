"""The CLI question prompt's answer deadline.

``ask_user_question`` hands the frontend a wall to answer within. In the terminal
that means stdin is polled rather than read with ``input()``, which cannot be
interrupted: a question raised while nobody is at the keyboard has to end by itself,
and it ends by telling the tool nobody answered — never by inventing a selection.
"""

from __future__ import annotations

import os
import sys
import unittest

from mimir.client.ui.cli.main import _cli_request_question

_QUESTION = {
    "question": "Which DB?",
    "header": "Database",
    "options": [{"label": "Postgres"}, {"label": "SQLite"}],
}


class _PipedStdin:
    """Replaces stdin with a pipe, so a poll can find it readable — or not."""

    def __enter__(self):
        self._read_fd, self._write_fd = os.pipe()
        self._saved = sys.stdin
        sys.stdin = os.fdopen(self._read_fd, "r")
        return self

    def send(self, line: str) -> None:
        os.write(self._write_fd, line.encode())

    def __exit__(self, *exc) -> None:
        sys.stdin.close()
        sys.stdin = self._saved
        os.close(self._write_fd)


class CliQuestionDeadlineTests(unittest.TestCase):
    def test_silence_ends_the_batch_as_timed_out(self) -> None:
        with _PipedStdin():
            result = _cli_request_question([_QUESTION], timeout_secs=0.1)

        self.assertEqual(result["answers"], [])
        self.assertTrue(result["timed_out"])

    def test_an_answer_within_the_wall_is_read(self) -> None:
        with _PipedStdin() as stdin:
            stdin.send("1\n")
            result = _cli_request_question([_QUESTION], timeout_secs=5)

        self.assertEqual(
            result["answers"], [{"selected": ["Postgres"], "other_text": None}]
        )
        self.assertNotIn("timed_out", result)

    def test_no_wall_means_no_poll(self) -> None:
        """Plan approval passes none, and reads with ``input()`` as it always has."""
        with _PipedStdin() as stdin:
            stdin.send("2\n")
            result = _cli_request_question([_QUESTION])

        self.assertEqual(
            result["answers"], [{"selected": ["SQLite"], "other_text": None}]
        )


if __name__ == "__main__":
    unittest.main()
