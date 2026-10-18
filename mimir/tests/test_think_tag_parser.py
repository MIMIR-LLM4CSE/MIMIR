"""The <think> tag state machine every streaming backend shares (no network)."""

import unittest

from mimir.client.llm.tag_parser import ThinkTagParser


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def parser(self) -> ThinkTagParser:
        return ThinkTagParser(
            token_callback=lambda t: self.events.append(("content", t)),
            think_token_callback=lambda t: self.events.append(("think", t)),
            think_start_callback=lambda: self.events.append(("start", "")),
            think_end_callback=lambda: self.events.append(("end", "")),
        )

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.events if k == kind)


class ThinkTagParserTests(unittest.TestCase):
    def test_a_closed_block_ends_once_even_when_finished(self) -> None:
        # vLLM fired the end callback itself after the parser already had: twice.
        rec = _Recorder()
        p = rec.parser()
        p.feed_content("<think>why</think>answer")
        p.finish()
        self.assertEqual(rec.count("start"), 1)
        self.assertEqual(rec.count("end"), 1)
        self.assertEqual("".join(p.content_parts), "answer")

    def test_native_reasoning_is_closed_by_finish(self) -> None:
        rec = _Recorder()
        p = rec.parser()
        p.feed_reasoning("step one")
        p.feed_reasoning(" step two")
        p.feed_content("answer")
        p.finish()
        self.assertEqual(rec.events[0], ("start", ""))
        self.assertEqual(rec.count("end"), 1)
        self.assertEqual("".join(p.thinking_parts), "step one step two")

    def test_a_tag_prefix_at_the_end_of_the_stream_is_not_lost(self) -> None:
        p = ThinkTagParser(token_callback=lambda _t: None)
        p.feed_content("x < y and a <")
        p.finish()
        self.assertEqual("".join(p.content_parts), "x < y and a <")

    def test_a_stray_closing_tag_split_across_chunks_does_not_leak(self) -> None:
        p = ThinkTagParser(token_callback=lambda _t: None)
        p.feed_content("answer</thi")
        p.feed_content("nk> done")
        p.finish()
        self.assertEqual("".join(p.content_parts), "answer done")

    def test_nothing_thought_means_no_end(self) -> None:
        rec = _Recorder()
        p = rec.parser()
        p.feed_content("plain")
        p.finish()
        self.assertEqual(rec.count("end"), 0)


if __name__ == "__main__":
    unittest.main()
