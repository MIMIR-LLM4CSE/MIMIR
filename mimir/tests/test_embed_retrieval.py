"""Tests for the shared embedding helper and its two consumers, focused on the
graceful-degradation contract: with no embedding backend everything must fall back
to the pre-existing lexical / substring behaviour (keeps the suite hermetic).
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SHARED = _ROOT / "mimir" / "servers" / "_shared"
for _p in (str(_SHARED),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mimir.servers._shared import embed
from mimir.client.query_engine import toollist


def _load_server_memory():
    spec = importlib.util.spec_from_file_location(
        "server_memory",
        _ROOT / "mimir" / "servers" / "agent_state" / "server_memory.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description}}


class EmbedHelperTests(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in
                     ("LLM_BACKEND", "MIMIR_EMBED_MODEL", "MIMIR_EMBED_BASE_URL")}
        embed._reset_availability_cache()

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        embed._reset_availability_cache()

    def test_embed_texts_empty_input_returns_empty(self):
        self.assertEqual(embed.embed_texts([]), [])

    def test_vllm_without_model_returns_none(self):
        os.environ["LLM_BACKEND"] = "vllm"
        os.environ.pop("MIMIR_EMBED_MODEL", None)
        self.assertIsNone(embed.embed_texts(["hello"]))

    def test_ray_without_model_returns_none(self):
        """Ray takes the OpenAI path, which cannot guess a served model name."""
        os.environ["LLM_BACKEND"] = "ray"
        os.environ.pop("MIMIR_EMBED_MODEL", None)
        self.assertIsNone(embed.embed_texts(["hello"]))
        self.assertIsNone(embed.embed_model_id())

    def test_ray_embeds_over_the_openai_path(self):
        os.environ["LLM_BACKEND"] = "ray"
        os.environ["MIMIR_EMBED_MODEL"] = "BAAI/bge-m3"
        seen = []
        original = embed._embed_vllm
        embed._embed_vllm = lambda texts, model: seen.append(model) or [[0.5]]
        try:
            self.assertEqual(embed.embed_texts(["hello"]), [[0.5]])
        finally:
            embed._embed_vllm = original
        self.assertEqual(seen, ["BAAI/bge-m3"])
        self.assertEqual(embed.embed_model_id(), "BAAI/bge-m3")

    def test_backend_exception_is_swallowed_to_none(self):
        os.environ["LLM_BACKEND"] = "ollama"
        original = embed._embed_ollama
        embed._embed_ollama = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        try:
            self.assertIsNone(embed.embed_texts(["hello"]))
        finally:
            embed._embed_ollama = original

    def test_is_available_false_when_backend_down(self):
        original = embed.embed_texts
        embed.embed_texts = lambda texts: None
        embed._reset_availability_cache()
        try:
            self.assertFalse(embed.is_available())
        finally:
            embed.embed_texts = original
            embed._reset_availability_cache()

    def test_lexical_rank_orders_by_overlap_then_index(self):
        ranked = embed.lexical_rank(
            "cluster job submission",
            ["file editor tool", "submit a cluster job to the queue", "cluster status"],
        )
        # index 1 has the most overlap, then 2, then 0 (zero overlap, last).
        self.assertEqual([i for i, _ in ranked], [1, 2, 0])

    def test_cosine_rank_ranks_identical_vector_first(self):
        ranked = embed.cosine_rank([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        self.assertEqual(ranked[0][0], 1)


def _fake_vec(text: str, vocab: list[str]) -> list[float]:
    """Deterministic bag-of-words vector over a fixed vocabulary — lets us exercise
    the semantic path (cosine selection, backfill, caching) without a live backend."""
    low = text.lower()
    return [1.0 if w in low else 0.0 for w in vocab]


class SemanticPathTests(unittest.TestCase):
    """Drive the embedding path with a fake embedder to prove the wiring: query and
    candidates are embedded, cosine-ranked, and the right items selected.

    The tool-list cap used to be the other caller of this path; it was removed, so the
    remaining consumer is memory search."""""

    VOCAB = ["cluster", "job", "slurm", "date", "benchmark", "performance"]

    def _fake_embed_texts(self, texts):
        return [_fake_vec(t, self.VOCAB) for t in texts]

    def test_memory_search_semantic_ranks_and_scores(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mem = _load_server_memory()
        d = Path(tmp.name)
        mem.MEMORY_DIR = str(d)
        mem.INDEX_FILE = str(d / "MEMORY.md")
        mem.EMBEDDINGS_FILE = str(d / "embeddings.json")
        mem._embed.is_available = lambda: True
        mem._embed.embed_texts = self._fake_embed_texts
        mem._embed.embed_one = lambda t: _fake_vec(t, self.VOCAB)
        mem._embed.embed_model_id = lambda: "fake-model"

        mem.memory_add("cluster job slurm submission notes", description="cluster jobs")
        mem.memory_add("date and time conversion", description="dates")
        # Vectors persisted on add.
        self.assertTrue(os.path.exists(mem.EMBEDDINGS_FILE))

        res = mem.memory_search("how do I submit a cluster job", limit=1)
        self.assertEqual(res["count"], 1)
        top = res["results"][0]
        self.assertIn("cluster", top["text"])
        self.assertIsNotNone(top["score"])  # semantic path attaches a score


class MemorySearchFallbackTests(unittest.TestCase):
    """With embeddings unavailable, memory_search keeps its substring semantics."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = _load_server_memory()
        d = Path(self._tmp.name)
        self.mem.MEMORY_DIR = str(d)
        self.mem.INDEX_FILE = str(d / "MEMORY.md")
        self.mem.EMBEDDINGS_FILE = str(d / "embeddings.json")
        # Force the lexical fallback path.
        self.mem._embed.is_available = lambda: False

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_then_substring_search(self):
        self.mem.memory_add("The vLLM backend is the most used in practice",
                            description="vllm backend usage")
        res = self.mem.memory_search("vLLM backend")
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["count"], 1)
        self.assertTrue(any("vLLM" in r["text"] for r in res["results"]))

    def test_no_embeddings_file_written_when_backend_down(self):
        self.mem.memory_add("some fact about tool limits", description="tool limits")
        self.assertFalse(os.path.exists(self.mem.EMBEDDINGS_FILE))

    def test_substring_miss_returns_empty(self):
        self.mem.memory_add("alpha beta gamma", description="greek letters")
        res = self.mem.memory_search("delta epsilon")
        self.assertEqual(res["count"], 0)

    def test_tag_filter_respected(self):
        self.mem.memory_add("fact one", description="one", tags=["keep"])
        self.mem.memory_add("fact two", description="two", tags=["other"])
        res = self.mem.memory_search("fact", tag="keep")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["results"][0]["tags"], ["keep"])


if __name__ == "__main__":
    unittest.main()
