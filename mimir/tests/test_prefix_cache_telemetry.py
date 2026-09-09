"""The vLLM prefix-cache report: the only direct evidence the cache is working.

It matters because the whole advertised tool list is now sent on every call — the
largest constant block in the prompt, and precisely what prefix caching should be
paying for. Without this report a silent prefix invalidator (a varying system prompt,
a reordered tool list, or a server started without `--enable-prefix-caching`) costs
that on every call with nothing to show for it.
"""
import logging
import unittest

from mimir.client.query_engine.backends import vllm_backend as vb


class _Details:
    def __init__(self, cached): self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt, cached, completion=7):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached)


class CacheUsageReportTests(unittest.TestCase):
    def _log(self, usage, level=logging.INFO):
        with self.assertLogs(vb.logger, level=level) as caught:
            vb.logger.info("probe")           # guarantees the block is non-empty
            vb._log_cache_usage(usage)
        return [r.getMessage() for r in caught.records]

    def test_reports_the_hit_rate(self):
        lines = self._log(_Usage(prompt=1000, cached=800))
        self.assertTrue(any("hit_rate=0.80" in m for m in lines))
        self.assertTrue(any("cached=800" in m for m in lines))

    def test_a_cold_prefix_reports_zero(self):
        # The shape that says something is invalidating the prefix on every call.
        self.assertTrue(any("hit_rate=0.00" in m for m in self._log(_Usage(1000, 0))))

    def test_no_usage_block_is_not_an_error(self):
        # A server that reports nothing must not break the turn.
        self.assertEqual(self._log(None), ["probe"])

    def test_a_malformed_usage_block_is_not_an_error(self):
        class _Broken:
            @property
            def prompt_tokens(self): raise RuntimeError("boom")
        self.assertEqual(self._log(_Broken()), ["probe"])

    def test_missing_details_reads_as_no_cache_rather_than_failing(self):
        class _NoDetails:
            prompt_tokens = 500
            completion_tokens = 1
            prompt_tokens_details = None
        self.assertTrue(any("cached=0" in m for m in self._log(_NoDetails())))

    def test_silent_when_nobody_is_listening(self):
        vb.logger.setLevel(logging.WARNING)
        self.addCleanup(vb.logger.setLevel, logging.NOTSET)
        with self.assertLogs(vb.logger, level=logging.WARNING) as caught:
            vb.logger.warning("probe")
            vb._log_cache_usage(_Usage(1000, 800))
        self.assertEqual([r.getMessage() for r in caught.records], ["probe"])


class StreamOptionsFallbackTests(unittest.TestCase):
    """Asking for usage must never cost a turn on a server that rejects the option."""

    class _Rejects:
        """Refuses `stream_options` once, like an OpenAI-compatible server that lacks it."""
        def __init__(self): self.calls = []
        class _Err(Exception):
            status_code = 400
            def __str__(self): return "unrecognized request argument: stream_options"
        @property
        def chat(self): return self
        @property
        def completions(self): return self
        def create(self, **kw):
            self.calls.append(dict(kw))
            if "stream_options" in kw:
                raise StreamOptionsFallbackTests._Rejects._Err()
            return "answer"

    def setUp(self):
        vb._NO_STREAM_OPTIONS.discard("m")
        self.addCleanup(vb._NO_STREAM_OPTIONS.discard, "m")

    def test_the_rejection_is_absorbed_and_the_turn_succeeds(self):
        client = self._Rejects()
        out = vb._create(client, {"model": "m", "stream_options": {"include_usage": True}})
        self.assertEqual(out, "answer")
        self.assertEqual(len(client.calls), 2)          # rejected, then retried without
        self.assertNotIn("stream_options", client.calls[1])

    def test_it_is_remembered_so_the_400_is_paid_once(self):
        vb._create(self._Rejects(), {"model": "m", "stream_options": {"include_usage": True}})
        client = self._Rejects()
        vb._create(client, {"model": "m", "stream_options": {"include_usage": True}})
        self.assertEqual(len(client.calls), 1)          # never sent the second time
        self.assertNotIn("stream_options", client.calls[0])


if __name__ == "__main__":
    unittest.main()
