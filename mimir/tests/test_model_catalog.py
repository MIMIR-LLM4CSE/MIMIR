"""The model catalog a sub-agent's model is picked from.

The orchestrator only sees what ``describe_served_models`` renders, once per session;
these tests pin what that text may claim and how a served id reaches its entry.
"""
from __future__ import annotations

import httpx

from mimir.client.config.model_catalog import (
    catalog_entry,
    decode_cost,
    describe_served_models,
    load_catalog,
    subagent_model_refusal,
)

# What the internal router listed on 2026-09-19.
_ROUTER = [
    {"id": "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"},
    {"id": "mistralai/Mistral-Medium-3.5-128B"},
    {"id": "Qwen/Qwen3-0.6B"},
    {"id": "RedHatAI/GLM-5.3-Flash-NVFP4"},
    {"id": "deepseek-ai/DeepSeek-V4.1-Flash"},
]


class TestMatching:
    def test_every_router_id_reaches_an_entry(self):
        for m in _ROUTER:
            assert catalog_entry(m["id"]), m["id"]

    def test_an_arbitrary_served_name_is_matched_through_its_root(self):
        assert not catalog_entry("fast")
        assert catalog_entry("fast", root="/models/deepseek-ai/DeepSeek-V4.1-Flash")["arch"] == "moe"

    def test_an_unknown_model_has_no_entry(self):
        assert catalog_entry("acme/Unknown-7B") == {}


class TestSpeed:
    def test_moe_counts_active_parameters_and_precision(self):
        assert decode_cost(catalog_entry("RedHatAI/GLM-5.3-Flash-NVFP4")) == 18 * 0.5
        assert decode_cost(catalog_entry("mistralai/Mistral-Medium-3.5-128B")) == 128 * 2

    def test_unknown_precision_is_taken_as_bf16(self):
        assert decode_cost({"active_params_b": 10}) == 20

    def test_no_active_parameters_means_no_estimate(self):
        assert decode_cost({}) is None


class TestDescription:
    def test_one_model_offers_nothing_else(self):
        text = describe_served_models(_ROUTER[3:4])
        assert "No other model" in text

    def test_a_not_delegable_model_does_not_count_as_a_choice(self):
        text = describe_served_models([_ROUTER[2], _ROUTER[3]])
        assert "No other model" in text

    def test_the_router_table_ranks_speed_among_usable_models_only(self):
        text = describe_served_models(_ROUTER)
        line = next(ln for ln in text.splitlines() if ln.startswith("- RedHatAI/GLM"))
        assert "speed 1/4" in line
        assert "Not for sub-agents: Qwen/Qwen3-0.6B" in text
        assert "- Qwen/" not in text

    def test_an_unmeasured_category_says_so(self):
        text = describe_served_models(_ROUTER)
        assert "instructions n/a" in text
        assert "Not measured comparably yet: instructions" in text

    def test_a_model_without_entry_is_still_offered(self):
        text = describe_served_models([_ROUTER[3], {"id": "acme/Unknown-7B"}])
        assert "- acme/Unknown-7B: speed ? | no benchmark data" in text

    def test_the_endpoint_window_wins_over_the_documented_one(self):
        text = describe_served_models([
            {"id": "RedHatAI/GLM-5.3-Flash-NVFP4", "max_model_len": 131072},
            _ROUTER[4]])
        assert "131k ctx" in text


class TestRefusal:
    def test_a_served_delegable_model_is_accepted(self):
        assert subagent_model_refusal("deepseek-ai/DeepSeek-V4.1-Flash", _ROUTER) is None

    def test_an_unserved_model_lists_only_usable_ones(self):
        why = subagent_model_refusal("gpt-oss", _ROUTER)
        assert "RedHatAI/GLM-5.3-Flash-NVFP4" in why and "Qwen3" not in why

    def test_nothing_listed_says_to_leave_it_empty(self):
        assert "lists no other model" in subagent_model_refusal("x", [])


class TestCatalogData:
    """Every number must be traceable: a score without a source is a guess."""

    def _entries(self):
        return {k: v for k, v in load_catalog().items() if not k.startswith("_")}

    def test_every_entry_has_sources_and_known_categories(self):
        categories = set(load_catalog()["_scale"]["categories"])
        weights = set(load_catalog()["_weight_bytes"])
        for name, entry in self._entries().items():
            assert entry.get("sources"), name
            assert set(entry["scores"]) == categories, name
            assert entry["weights"] in weights, name
            assert entry["arch"] in ("dense", "moe"), name
            if entry["arch"] == "dense":
                assert entry["active_params_b"] == entry["total_params_b"], name


def test_the_vllm_backend_hands_over_whole_model_entries(monkeypatch):
    from mimir.client.query_engine.backends.vllm_backend import VllmBackend

    monkeypatch.setenv("VLLM_BASE_URL", "https://endpoint.internal")

    def _get(self, url, *a, **kw):
        return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [
            {"id": "served-name", "root": "/models/org/Model-7B", "max_model_len": 8192},
            {"object": "model"},
        ]})

    monkeypatch.setattr(httpx.Client, "get", _get)
    assert VllmBackend().served_model_info() == [
        {"id": "served-name", "root": "/models/org/Model-7B", "max_model_len": 8192}]
