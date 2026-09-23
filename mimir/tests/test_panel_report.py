"""The scientific-computing panel: who fills it, and how the client asks.

The panel shows what is specific to this work — the optimisation in progress, the
machine, the sub-agents, the runs still going. Two of those sections are the client's
own; the rest are filled by the servers that own the facts, asked by capability so a
plugin server's section appears without a client edit.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "hpc", SERVERS_DIR / "proxy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    """Load one server module by path, as its own subprocess would."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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


def test_a_server_with_nothing_to_say_gets_no_heading():
    """A section is a heading plus what is under it.

    A cluster section on a laptop has no queue, no partitions and no nodes — it returns
    neither lines nor detail, and rendering its title anyway gives the user a heading to
    read in order to learn there is nothing to read. A section that reports *about*
    having nothing ("no optimisation session in this workspace") says so in `detail`,
    and still appears.
    """
    import asyncio
    import json
    from mimir.client.ui.ws.ws_worker import _AgentWorker

    payloads = {
        "empty": {"title": "Cluster", "lines": []},
        "spoken_for": {"title": "Optimisation", "lines": [],
                       "detail": "No optimisation session in this workspace."},
    }

    class _Session:
        def __init__(self, tool):
            self.tool = tool

        async def call_tool(self, tool, args):
            return self.tool

    agent = types.SimpleNamespace(
        tool_caps=_caps(empty={"section": "Cluster", "order": 10},
                        spoken_for={"section": "Optimisation", "order": 20}),
        tool_owner={"empty": "hpc", "spoken_for": "proxy"},
        sessions={"hpc": _Session("empty"), "proxy": _Session("spoken_for")},
        _normalize_tool_content=lambda raw: json.dumps(payloads[raw]),
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


def test_the_sections_read_from_the_work_outwards():
    """Optimisation, then the cluster it runs on, then the host MIMIR itself sits on.

    The order is the order of the question: what is being optimised, where it will run,
    and — last, because on a cluster it is the one machine nothing is measured on —
    this host.
    """
    from mimir.tests import _golden_caps as golden
    declared = panel_sections(golden.build_declared_registry())
    assert [spec["section"] for _, spec in declared] == [
        "Optimisation", "Cluster", "Machine"]


def test_a_host_without_a_scheduler_reports_no_cluster_at_all():
    """Not an error and not a row saying "unavailable" — nothing.

    The client drops a section with neither lines nor detail, so a laptop does not grow
    an empty cluster panel.
    """
    hpc = _load("server_hpc_panel", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    with mock.patch.object(hpc, "_run_bash",
                           return_value={"status": "error", "error": "no sinfo here"}):
        out = hpc.hpc_panel_report()
    assert out["status"] == "ok"
    assert out["lines"] == []
    assert not out.get("detail")


def test_the_queue_and_the_room_are_what_the_cluster_section_says():
    hpc = _load("server_hpc_panel2", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()

    def _bash(cmd, timeout):
        if cmd.startswith("squeue"):
            return {"status": "ok", "stdout": "1|me|RUNNING|1:00|2:00|1|n1|a\n"
                                              "2|me|PENDING|0:00|2:00|1|(Prio)|b\n"}
        return {"status": "ok", "stdout": "gpu|up|1-00|4|idle|32|64000\n"
                                          "gpu|up|1-00|6|alloc|32|64000\n"}

    with mock.patch.object(hpc, "_run_bash", _bash), \
            mock.patch.object(hpc, "_profiled_node_lines", return_value=[]):
        out = hpc.hpc_panel_report()
    values = {line["label"]: line["value"] for line in out["lines"]}
    assert values["your jobs"] == "1 pending, 1 running"
    # 4 idle of 10: whether a job submitted now starts now is the whole question.
    assert values["gpu"] == "4/10 nodes idle"


def _proxy():
    return _load("server_proxy_panel", SERVERS_DIR / "proxy" / "server_proxy.py")


def test_a_session_pinned_elsewhere_says_so_before_the_run_not_after():
    """The "incomparable" verdict arrives after the build, the measurement and the
    wait. The same comparison costs nothing before any of it."""
    proxy = _proxy()
    with mock.patch.object(proxy, "_this_machine",
                           return_value={"machine_signature": "here", "host": "login01"}):
        lines = proxy._comparability_lines(
            {"machine": {"machine_signature": "elsewhere", "host": "node042"}}, "time_s")
    assert lines[0] == {"label": "measured on", "value": "node042"}
    assert lines[1]["state"] == "warn"
    assert "incomparable" in lines[1]["value"]


def test_a_session_pinned_here_raises_nothing():
    proxy = _proxy()
    with mock.patch.object(proxy, "_this_machine",
                           return_value={"machine_signature": "here", "host": "node042"}):
        lines = proxy._comparability_lines(
            {"machine": {"machine_signature": "here", "host": "node042"}}, "time_s")
    assert len(lines) == 1
    assert "this machine" in lines[0]["value"]


def test_a_metric_that_does_not_depend_on_the_machine_is_never_pinned():
    """An accuracy is reproducible anywhere; warning about the host would be noise."""
    proxy = _proxy()
    assert proxy._comparability_lines(
        {"machine": {"machine_signature": "elsewhere", "host": "node042"}},
        "l2_error") == []


def test_nothing_is_said_until_a_baseline_has_pinned_a_machine():
    proxy = _proxy()
    assert proxy._comparability_lines({}, "time_s") == []
