"""The scientific-computing panel: who fills it, and how the client asks.

The panel shows what is specific to this work — the optimisation in progress, the
machine, the sub-agents, the runs still going. Two of those sections are the client's
own; the rest are filled by the servers that own the facts, asked by capability so a
plugin server's section appears without a client edit.
"""
from __future__ import annotations

import types

from mimir.client.context.capabilities import PANEL_REPORT, ToolCaps, panel_sections


def _caps(**specs) -> dict[str, ToolCaps]:
    return {
        name: ToolCaps(name=name, capabilities=frozenset({PANEL_REPORT}), panel=spec)
        for name, spec in specs.items()
    }


class TestDeclaredSections:
    def test_sections_come_back_in_the_order_they_declared(self):
        caps = _caps(
            machine={"section": "Machine", "order": 40},
            optimisation={"section": "Optimisation", "order": 20},
        )
        assert [name for name, _ in panel_sections(caps)] == ["optimisation", "machine"]

    def test_a_tool_without_the_capability_fills_nothing(self):
        caps = {"other": ToolCaps(name="other", panel={"section": "Machine"})}
        assert panel_sections(caps) == []

    def test_a_declaration_without_a_section_is_ignored(self):
        caps = _caps(vague={"order": 10})
        assert panel_sections(caps) == []

    def test_the_call_arguments_travel_with_the_section(self):
        caps = _caps(status={"section": "Optimisation", "args": {"op": "status"}})
        assert panel_sections(caps)[0][1]["args"] == {"op": "status"}


class TestTheServersThatFillIt:
    """The two first-party sections, read off the real declarations."""

    def _declared(self):
        from mimir.tests import _golden_caps as golden
        return golden.build_declared_registry()

    def test_the_proxy_and_the_platform_each_own_a_section(self):
        sections = dict(panel_sections(self._declared()))
        by_section = {spec["section"] for spec in sections.values()}
        assert {"Optimisation", "Machine"} <= by_section

    def test_every_panel_tool_is_read_only(self):
        """The panel is a drawer the user opens, not a step the agent takes."""
        declared = self._declared()
        for name, _ in panel_sections(declared):
            caps = declared[name].capabilities
            assert "plan_blocked" not in caps and "code_exec" not in caps, name


def test_the_worker_asks_by_capability_and_survives_a_silent_server(monkeypatch):
    """One server that cannot answer costs its own section, never the panel."""
    import asyncio
    import json
    from mimir.client.ui.ws.ws_worker import _AgentWorker

    class _Session:
        def __init__(self, payload=None, boom=False):
            self.payload, self.boom = payload, boom

        async def call_tool(self, tool, args):
            if self.boom:
                raise RuntimeError("that server is gone")
            return self.payload

    agent = types.SimpleNamespace(
        tool_caps=_caps(good={"section": "Optimisation", "order": 10},
                        bad={"section": "Machine", "order": 20}),
        tool_owner={"good": "proxy", "bad": "platform"},
        sessions={"proxy": _Session(payload="x"), "platform": _Session(boom=True)},
        _normalize_tool_content=lambda raw: json.dumps(
            {"title": "Optimisation", "lines": [{"label": "state", "value": "running"}]}),
    )
    loop = asyncio.new_event_loop()
    worker = object.__new__(_AgentWorker)
    worker._agent = agent
    worker._loop = loop

    future = worker.panel_sections()
    loop.run_until_complete(asyncio.sleep(0))
    sections = future.result(timeout=5)
    loop.close()

    assert [s["section"] for s in sections] == ["Optimisation"]
    assert sections[0]["lines"] == [{"label": "state", "value": "running"}]
