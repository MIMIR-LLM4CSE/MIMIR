"""The scientific-computing panel: who fills it, and how the client asks.

The panel shows what is specific to this work — the optimisation in progress, the
machine, the sub-agents, the runs still going. Two of those sections are the client's
own; the rest are filled by the servers that own the facts, asked by capability so a
plugin server's section appears without a client edit.
"""
from __future__ import annotations

from mimir.tests.support.modules import spec_for
import asyncio
import importlib.util
import os
import json
import types
from pathlib import Path
from unittest import mock
from mimir.tests.support.agents import fake_agent


SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"


def _load(name: str, path: Path):
    """Load one server module by path, as its own subprocess would."""
    spec = spec_for(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(hpc, **kwargs) -> dict:
    return asyncio.run(hpc.hpc_panel_report(**kwargs))

from mimir.servers._shared import slurm_site

from mimir.client.tools.capabilities import PANEL_REPORT, ToolCaps, panel_sections
from mimir.client.core.turn_state import TurnState


def _mcp(payload: dict) -> types.SimpleNamespace:
    """A tool result as an MCP session returns it: one text block carrying the JSON."""
    return types.SimpleNamespace(content=[types.SimpleNamespace(text=json.dumps(payload))])


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

    def test_a_caching_tool_names_the_argument_that_bypasses_its_cache(self):
        """So the client can tell a reader's press apart from a timer's tick."""
        caps = _caps(cluster={"section": "Cluster", "refresh_arg": "refresh"})
        assert panel_sections(caps)[0][1]["refresh_arg"] == "refresh"


class TestTheServersThatFillIt:
    """The two first-party sections, read off the real declarations."""

    def _declared(self):
        from mimir.tests.support import golden_caps as golden
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


def _drain(loop, future):
    """Turn the loop until the panel's coroutine is done.

    It asks every server at once, so it yields once per section rather than once in
    total; a single tick of the loop would read a half-filled list.
    """
    import asyncio
    while not future.done():
        loop.run_until_complete(asyncio.sleep(0))


def test_the_worker_asks_by_capability_and_survives_a_silent_server(monkeypatch):
    """One server that cannot answer costs its own section, never the panel."""
    import asyncio
    from mimir.client.ui.ws.worker.worker import AgentWorker

    class Connection:
        def __init__(self, payload=None, boom=False):
            self.payload, self.boom = payload, boom

        async def call_tool(self, tool, args):
            if self.boom:
                raise RuntimeError("that server is gone")
            return _mcp({"title": "Optimisation",
                         "lines": [{"label": "state", "value": "running"}]})

    agent = fake_agent(
        tool_caps=_caps(good={"section": "Optimisation", "order": 10},
                        bad={"section": "Machine", "order": 20}),
        tool_owner={"good": "proxy", "bad": "platform"},
        sessions={"proxy": Connection(payload="x"), "platform": Connection(boom=True)},turn=TurnState())
    loop = asyncio.new_event_loop()
    worker = AgentWorker("test-model", start=False)
    worker._agent = agent
    worker._loop = loop

    future = worker.panel.sections()
    _drain(loop, future)
    sections = future.result(timeout=5)
    loop.close()

    assert [s["section"] for s in sections] == ["Optimisation"]
    assert sections[0]["lines"] == [{"label": "state", "value": "running"}]


def test_the_servers_are_all_asked_before_any_of_them_answers():
    """Together, not in turn — and still in the declared order on the way out.

    Each section is owned by a different process, and the slow ones are slow because
    they are waiting on a scheduler. Asked one after another the drawer cost the sum
    of those waits; asked at once it costs the longest of them.
    """
    import asyncio
    from mimir.client.ui.ws.worker.worker import AgentWorker

    started: list[str] = []
    release = []

    class Connection:
        async def call_tool(self, tool, args):
            started.append(tool)
            await release[0]          # nobody answers until everyone has asked
            return _mcp({"lines": [{"label": "l", "value": "v"}]})

    agent = fake_agent(
        tool_caps=_caps(machine={"section": "Machine", "order": 10},
                        cluster={"section": "Cluster", "order": 20},
                        proxy={"section": "Optimisation", "order": 30}),
        tool_owner={"machine": "platform", "cluster": "hpc", "proxy": "proxy"},
        sessions={"platform": Connection(), "hpc": Connection(), "proxy": Connection()},turn=TurnState())
    loop = asyncio.new_event_loop()
    release.append(loop.create_future())
    worker = AgentWorker("test-model", start=False)
    worker._agent = agent
    worker._loop = loop

    future = worker.panel.sections()
    while len(started) < 3:
        loop.run_until_complete(asyncio.sleep(0))
    release[0].set_result(None)
    _drain(loop, future)
    sections = future.result(timeout=5)
    loop.close()

    assert started == ["machine", "cluster", "proxy"]
    assert [s["section"] for s in sections] == ["Machine", "Cluster", "Optimisation"]


def test_only_the_readers_own_refresh_reaches_past_a_servers_cache(monkeypatch):
    """A section that costs an ``sinfo`` is served from a cache, which is what makes a
    panel on a timer affordable — and what made pressing ⟳ hand back the page just
    read. The tool that caches names the argument meaning "read it again"; the client
    sets it for that press and for nothing else.
    """
    import asyncio
    from mimir.client.ui.ws.worker.worker import AgentWorker

    seen: list[dict] = []

    class Connection:
        async def call_tool(self, tool, args):
            seen.append({"tool": tool, **args})
            return _mcp({"lines": [{"label": "l", "value": "v"}]})

    agent = fake_agent(
        tool_caps=_caps(cluster={"section": "Cluster", "order": 10,
                                 "refresh_arg": "refresh"},
                        machine={"section": "Machine", "order": 20}),
        tool_owner={"cluster": "hpc", "machine": "platform"},
        sessions={"hpc": Connection(), "platform": Connection()},turn=TurnState())
    loop = asyncio.new_event_loop()
    worker = AgentWorker("test-model", start=False)
    worker._agent = agent
    worker._loop = loop

    for force in (False, True):
        future = worker.panel.sections(force=force)
        _drain(loop, future)
        future.result(timeout=5)
    loop.close()

    assert seen == [
        {"tool": "cluster"},
        {"tool": "machine"},
        {"tool": "cluster", "refresh": True},
        # Never invented for a tool that did not declare it: an unexpected argument
        # costs that server its whole section.
        {"tool": "machine"},
    ]


def test_a_refresh_drops_the_digest_that_outlives_every_other_cache():
    """The one cache a timer can never clear.

    A scheduler's digest is memoised for the life of the server process, so a cluster
    that was unreachable at startup — or a partition added since — stays invisible
    until something drops it. That is what the reader's ⟳ is for.
    """
    hpc = _load("server_hpc_refresh", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    forgotten: list = []

    with mock.patch.object(hpc, "_forget_cluster_digest",
                           side_effect=lambda *a: forgotten.append(a)), \
         mock.patch.object(hpc, "slurm_hosts", return_value=[]), \
         mock.patch.object(hpc, "_queue_line", return_value=None), \
         mock.patch.object(hpc, "_partition_lines", return_value=[]):
        with mock.patch.object(hpc, "_host_tree", return_value=[{"id": "host:stale"}]):
            _report(hpc)
        with mock.patch.object(hpc, "_host_tree", return_value=[{"id": "host:fresh"}]):
            held = _report(hpc)
            assert held["groups"] == [{"id": "host:stale"}]      # the timer's answer
            asked = _report(hpc, refresh=True)

    assert asked["groups"] == [{"id": "host:fresh"}]
    assert forgotten == [()]                                  # every host, not one


def test_a_profile_landing_shows_without_a_refresh(tmp_path):
    """A node profiled during a turn reaches the timer's answer, not only ⟳'s."""
    hpc = _load("server_hpc_profile_lands", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    host_dir = tmp_path / "local"
    host_dir.mkdir()

    with mock.patch.object(hpc, "slurm_hosts", return_value=[""]), \
         mock.patch.object(hpc, "profile_host_dir", return_value=str(host_dir)), \
         mock.patch.object(hpc, "_queue_line", return_value=None), \
         mock.patch.object(hpc, "_partition_lines", return_value=[]):
        with mock.patch.object(hpc, "_host_tree", return_value=[{"id": "before"}]):
            assert _report(hpc)["groups"] == [{"id": "before"}]
        (host_dir / "n01.json").write_text("{}")
        os.utime(host_dir, ns=(0, os.stat(host_dir).st_mtime_ns + 10**9))
        with mock.patch.object(hpc, "_host_tree", return_value=[{"id": "after"}]):
            assert _report(hpc)["groups"] == [{"id": "after"}]


def test_a_host_that_did_not_answer_is_asked_again_soon():
    hpc = _load("server_hpc_retry", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    with mock.patch.object(hpc, "slurm_hosts", return_value=[]), \
         mock.patch.object(hpc, "_queue_line", return_value=None), \
         mock.patch.object(hpc, "_partition_lines", return_value=[]):
        with mock.patch.object(hpc, "_host_tree",
                               return_value=[{"id": "h", "state": "warn"}]):
            _report(hpc)
        hpc._PANEL_CACHE["hosts"]["at"] -= hpc._PANEL_RETRY_SECS + 1
        with mock.patch.object(hpc, "_host_tree", return_value=[{"id": "h"}]):
            assert _report(hpc)["groups"] == [{"id": "h"}]


def test_a_server_with_nothing_to_say_gets_no_heading():
    """A section with neither lines nor detail gets no heading; one that says it has nothing
    (in ``detail``) still appears.
    """
    import asyncio
    from mimir.client.ui.ws.worker.worker import AgentWorker

    payloads = {
        "empty": {"title": "Cluster", "lines": []},
        "spoken_for": {"title": "Optimisation", "lines": [],
                       "detail": "No optimisation session in this workspace."},
    }

    class Connection:
        def __init__(self, tool):
            self.tool = tool

        async def call_tool(self, tool, args):
            return _mcp(payloads[self.tool])

    agent = fake_agent(
        tool_caps=_caps(empty={"section": "Cluster", "order": 10},
                        spoken_for={"section": "Optimisation", "order": 20}),
        tool_owner={"empty": "hpc", "spoken_for": "proxy"},
        sessions={"hpc": Connection("empty"), "proxy": Connection("spoken_for")},turn=TurnState())
    loop = asyncio.new_event_loop()
    worker = AgentWorker("test-model", start=False)
    worker._agent = agent
    worker._loop = loop

    future = worker.panel.sections()
    _drain(loop, future)
    sections = future.result(timeout=5)
    loop.close()

    assert [s["section"] for s in sections] == ["Optimisation"]


def test_a_section_that_only_has_groups_is_kept():
    """A tree is something to say, even with no lines beside it.

    The drop rule exists so a facility this machine does not have grows no heading; a
    server answering with a nested shape and nothing flat is the opposite case.
    """
    import asyncio
    from mimir.client.ui.ws.worker.worker import AgentWorker

    payload = {"title": "Cluster", "lines": [],
               "groups": [{"id": "host:local", "label": "local"}]}

    class Connection:
        async def call_tool(self, tool, args):
            return _mcp(payload)

    agent = fake_agent(
        tool_caps=_caps(tree={"section": "Cluster", "order": 10}),
        tool_owner={"tree": "hpc"},
        sessions={"hpc": Connection()},turn=TurnState())
    loop = asyncio.new_event_loop()
    worker = AgentWorker("test-model", start=False)
    worker._agent = agent
    worker._loop = loop
    future = worker.panel.sections()
    _drain(loop, future)
    sections = future.result(timeout=5)
    loop.close()

    assert [s["section"] for s in sections] == ["Cluster"]
    # Passed through unread: this end renders what the server chose to say.
    assert sections[0]["groups"] == payload["groups"]


_DIGEST = {
    "scheduler": "hpc1", "built_at": "2026-10-13T00:00:00Z", "source": "scontrol",
    "signature": "abc", "degraded": "",
    "partitions": [{"partition": "gpu"}, {"partition": "cpu"}],
    "node_types": [{"arch": "x86_64", "cpus": 64, "mem_mb": 515000,
                    "gres": "gpu:a100:4", "sockets": 2, "cores_per_socket": 32,
                    "threads_per_core": 1, "features": "avx512,ib",
                    "partitions": ["gpu"], "example_nodes": ["gpu001", "gpu002"],
                    "nodes_total": 10}],
}

_SILENT_DIGEST = {"scheduler": "hpc2", "built_at": "", "source": "", "signature": "",
                  "degraded": "", "partitions": [], "node_types": []}


def _tree(hosts, digests, profiles=None):
    """Build the host tree against canned digests and profiles — no scheduler is asked."""
    hpc = _load("server_hpc_tree", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    with mock.patch.object(hpc, "slurm_hosts", return_value=hosts), \
         mock.patch.object(hpc, "_cluster_digest", side_effect=lambda h: digests[h]), \
         mock.patch.object(hpc, "read_node_profile",
                           side_effect=lambda node, host: (profiles or {}).get(node)):
        return hpc._host_tree()


def test_every_registered_scheduler_is_in_the_tree():
    """Including one nothing has been probed on yet.

    What you have registered is what you can submit to, so it is what the panel has to
    show — a cluster missing from the list reads as a cluster MIMIR cannot reach.
    """
    tree = _tree(["", "hpc1"], {"": _SILENT_DIGEST, "hpc1": _DIGEST})
    assert [g["label"] for g in tree] == ["local", "hpc1"]
    assert [g["local"] for g in tree] == [True, False]


def test_the_schedulers_are_all_reached_at_once():
    """A digest this workspace does not hold yet is an ssh away, and the wait is the
    cluster's, not this process's. Three of them one after another is three waits for
    a reader who asked one question."""
    import threading
    hpc = _load("server_hpc_parallel", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    hosts = ["hpc1", "hpc2", "hpc3"]
    barrier = threading.Barrier(len(hosts), timeout=5)
    probed: set[str] = set()

    def _slow(host):
        if host not in probed:        # the first ask per host is the one that reaches
            barrier.wait()            # raises unless all three are inside at once
            probed.add(host)
        return _DIGEST

    with mock.patch.object(hpc, "slurm_hosts", return_value=hosts), \
         mock.patch.object(hpc, "_cluster_digest", side_effect=_slow), \
         mock.patch.object(hpc, "read_node_profile", return_value=None):
        tree = hpc._host_tree()
    assert [g["label"] for g in tree] == hosts


def test_a_silent_host_is_listed_as_unknown_not_as_empty():
    """An unreachable cluster and a cluster with no machines are not the same thing,
    and only one of them is worth waiting for."""
    tree = _tree(["hpc2"], {"hpc2": _SILENT_DIGEST})
    assert tree[0]["state"] == "warn"
    assert tree[0]["value"] == "did not answer"
    assert "unknown, not absent" in tree[0]["lines"][-1]["value"]
    assert "groups" not in tree[0]        # never an empty machine list


def test_a_node_kind_is_named_by_its_nodes_and_lists_its_specs():
    tree = _tree(["hpc1"], {"hpc1": _DIGEST})
    kind = tree[0]["groups"][0]
    assert kind["label"] == "gpu001, gpu002 +8"
    assert kind["nodes"] == ["gpu001", "gpu002"]
    values = {line["label"]: line["value"] for line in kind["lines"]}
    assert values["arch"] == "x86_64"
    assert values["cpus"] == "64 (2 sockets × 32 cores × 1 threads)"
    assert values["gpu"] == "4× a100"
    assert values["cpu mem"] == "503 GB"
    assert values["gpu mem"] == "unknown until a node is profiled"
    assert values["partitions"] == "gpu"
    assert values["nodes"] == "10"


def test_an_unprofiled_kind_offers_the_request_that_would_fix_it_and_says_nothing_else():
    """An unprofiled kind shows the server-written request in its row, and no badge."""
    kind = _tree(["hpc1"], {"hpc1": _DIGEST})[0]["groups"][0]
    assert "state" not in kind
    assert "value" not in kind
    assert kind["action"]["kind"] == "probe"
    assert "'hpc1'" in kind["action"]["prompt"]
    assert "'gpu'" in kind["action"]["prompt"]


def test_a_profiled_kind_carries_what_only_the_node_could_say():
    """Slurm reports cores and memory but never the CPU model, the vector ISA or the
    name -march=native resolves to. That is the whole reason a probe exists."""
    profile = {"node": "gpu001", "probed_at": "2026-10-02", "partial": False,
               "profile": {"cpu": {"model": "AMD EPYC 7763",
                                   "simd": {"avx2": True, "avx512f": True}},
                           "march": {"native": "znver3"},
                           "os": {"id": "rhel", "version": "8.10", "glibc": "2.28"},
                           "gpu": {"devices": [{"name": "A100", "memory": "81920 MiB",
                                                "compute_cap": "8.0"}]}}}
    kind = _tree(["hpc1"], {"hpc1": _DIGEST}, {"gpu001": profile})[0]["groups"][0]
    values = {line["label"]: line["value"] for line in kind["lines"]}
    assert values["cpu"] == "AMD EPYC 7763"
    assert values["-march=native"] == "znver3"
    assert values["os"] == "rhel 8.10 · glibc 2.28"
    assert values["compute capability"] == "8.0"
    assert values["gpu mem"] == "80 GB per GPU"
    labels = [line["label"] for line in kind["lines"]]
    assert labels.index("gpu mem") == labels.index("cpu mem") + 1
    assert "Profiled on gpu001, 2026-10-02" in kind["detail"]
    assert "action" not in kind          # nothing left to ask for
    assert kind.get("state") != "warn"


def test_a_kinds_id_is_the_same_on_the_next_refresh():
    """The panel's open rows are keyed on it. A tree that re-identified its rows every
    poll would collapse under whoever was reading it."""
    first = _tree(["hpc1"], {"hpc1": _DIGEST})[0]["groups"][0]["id"]
    shuffled = {**_DIGEST, "node_types": [dict(reversed(list(
        _DIGEST["node_types"][0].items())))]}
    second = _tree(["hpc1"], {"hpc1": shuffled})[0]["groups"][0]["id"]
    assert first == second


def _sections_with_hosts():
    return [{"section": "Cluster", "title": "Cluster", "lines": [], "detail": "",
             "groups": [
                 {"id": "host:local", "label": "local", "host": "local", "local": True,
                  "groups": [{"id": "host:local/kind:aa", "label": "64 cpus",
                              "nodes": ["cpu001", "cpu002"]}]},
                 {"id": "host:hpc1", "label": "hpc1", "host": "hpc1", "local": False,
                  "groups": [{"id": "host:hpc1/kind:bb", "label": "4× a100",
                              "nodes": ["gpu001", "gpu007"]}]},
             ]}]


def test_recorded_clusters_are_shown_before_any_agent_is_up(tmp_path):
    """The registry is the workspace's; the first query should not be what reveals it."""
    from mimir.client.ui.ws.connection.panel_runs import recorded_hosts_sections
    assert recorded_hosts_sections(str(tmp_path)) == []
    (tmp_path / "slurm_hosts").write_text("hpc1\n\nhpc2\nhpc1\n")
    [section] = recorded_hosts_sections(str(tmp_path))
    assert section["section"] == "Cluster"
    assert [g["id"] for g in section["groups"]] == ["host:hpc1", "host:hpc2"]
    assert all(g["host"] == g["label"] for g in section["groups"])


def test_a_shell_run_lands_on_this_machine():
    """It has no scheduler to name, which is exactly what says it is here."""
    from mimir.client.ui.ws.connection.panel_runs import attach_runs
    run = {"job_key": "k1", "kind": "shell-command", "state": "running",
           "pid": 4242, "command": "make -j64", "origin": {"kind": "main"}}
    sections, unplaced = attach_runs(_sections_with_hosts(), [run])
    hosts = {g["label"]: g for g in sections[0]["groups"]}
    assert [b["job_key"] for b in hosts["local"]["badges"]] == ["k1"]
    assert "badges" not in hosts["hpc1"]
    assert hosts["local"]["badges"][0]["pid"] == 4242
    assert unplaced == []


def test_a_cluster_run_lands_on_its_host_and_on_its_node():
    """Two questions, two places: which cluster is busy, and which machine is."""
    from mimir.client.ui.ws.connection.panel_runs import attach_runs
    run = {"job_key": "15306620", "kind": "slurm-batch", "state": "running",
           "scheduler": "hpc1", "nodes": ["gpu007"], "job_id": "15306620"}
    sections, unplaced = attach_runs(_sections_with_hosts(), [run])
    hosts = {g["label"]: g for g in sections[0]["groups"]}
    assert [b["job_key"] for b in hosts["hpc1"]["badges"]] == ["15306620"]
    assert [b["job_key"] for b in hosts["hpc1"]["groups"][0]["badges"]] == ["15306620"]
    assert "badges" not in hosts["local"]
    assert unplaced == []


def test_a_run_on_a_node_outside_the_sample_still_lands_on_its_host():
    """A node kind lists an example of its nodes, not all of them.

    The host row is the one that must never miss a run; which kind of machine it got is
    a detail the run itself carries.
    """
    from mimir.client.ui.ws.connection.panel_runs import attach_runs
    run = {"job_key": "j9", "state": "running", "scheduler": "hpc1",
           "nodes": ["gpu042"]}
    sections, _ = attach_runs(_sections_with_hosts(), [run])
    hpc1 = sections[0]["groups"][1]
    assert [b["job_key"] for b in hpc1["badges"]] == ["j9"]
    assert "badges" not in hpc1["groups"][0]


def test_a_run_on_an_unknown_scheduler_is_still_shown_somewhere():
    """Unplaced, not dropped: a run the user cannot see is the worse failure."""
    from mimir.client.ui.ws.connection.panel_runs import attach_runs
    run = {"job_key": "j7", "state": "running", "scheduler": "hpc9"}
    sections, unplaced = attach_runs(_sections_with_hosts(), [run])
    assert all("badges" not in g for g in sections[0]["groups"])
    assert [r["job_key"] for r in unplaced] == ["j7"]


def test_a_badge_says_which_agent_launched_the_run():
    """A job on a node is worth little to read if it does not say whose."""
    from mimir.client.ui.ws.connection.panel_runs import attach_runs
    origin = {"kind": "subagent", "label": "port the solver to AVX-512",
              "session": "parent/subagents/sub-1a2b"}
    run = {"job_key": "k1", "state": "running", "origin": origin}
    sections, _ = attach_runs(_sections_with_hosts(), [run])
    assert sections[0]["groups"][0]["badges"][0]["origin"] == origin


def test_the_worker_reports_what_its_poll_learned():
    """``watched_runs`` keeps the status the watcher already held.

    Where a run is and how far it has got were read on every tick and then dropped on
    the way out, so the panel could only ever say that something was running.
    """
    from mimir.client.ui.ws.worker.jobs import _Watch
    from mimir.client.ui.ws.worker.worker import AgentWorker

    class _Task:
        @staticmethod
        def done():
            return False

    worker = AgentWorker("test-model", start=False)

    worker.jobs.watches = {"15306620": _Watch(
        _Task(),
        {"job_key": "15306620", "kind": "slurm-batch", "server": "hpc",
         "origin": {"kind": "main", "label": "", "session": "s1"}},
    )}
    worker.jobs.watches["15306620"].status = {
        "state": "running", "phase": "linking", "percent": 62,
        "scheduler": "hpc1", "nodes": ["gpu007"], "at": 1234.5,
    }
    run = worker.watched_runs()[0]
    assert run["nodes"] == ["gpu007"]
    assert run["scheduler"] == "hpc1"
    assert run["percent"] == 62
    assert run["origin"]["session"] == "s1"
    # A monotonic clock reading means nothing outside this process.
    assert "at" not in run


def test_the_sections_read_from_the_machine_outwards():
    """This host, then the clusters it reaches, then what is being optimised on them.

    Nearest first. The machine section is the one that is always there and always cheap,
    so the drawer says something true before any scheduler has been asked anything; the
    optimisation is last because it is the section a workspace most often has none of.
    """
    from mimir.tests.support import golden_caps as golden
    declared = panel_sections(golden.build_declared_registry())
    assert [spec["section"] for _, spec in declared] == [
        "Machine", "Cluster", "Optimisation"]


def test_a_host_without_a_scheduler_reports_no_cluster_at_all():
    """Not an error and not a row saying "unavailable" — nothing.

    The client drops a section with neither lines nor detail, so a laptop does not grow
    an empty cluster panel.
    """
    hpc = _load("server_hpc_panel", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()
    # Both halves have to be silent for the section to be. The scheduler inventory is
    # its own question — a workspace can have registered a cluster over ssh while the
    # machine the panel runs on has no Slurm at all — so it is emptied explicitly
    # rather than left to whatever this test host happens to have installed.
    with mock.patch.object(hpc.cli, "run_bash",
                           return_value={"status": "error", "error": "no sinfo here"}), \
         mock.patch.object(hpc, "slurm_hosts", return_value=[]):
        out = _report(hpc)
    assert out["status"] == "ok"
    assert out["lines"] == []
    assert not out.get("groups")
    assert not out.get("detail")


def test_the_queue_and_the_room_are_what_the_cluster_section_says():
    hpc = _load("server_hpc_panel2", SERVERS_DIR / "hpc" / "server_hpc.py")
    hpc._PANEL_CACHE.clear()

    # Every read names the scheduler it went to, so the stub takes the host and one
    # scheduler is forced: the probe is a fact about the developer's machine, and the
    # panel asks the real resolution, which with nothing reachable refuses.
    def _bash(cmd, timeout, stdin=None, *, host=""):
        if cmd.startswith("squeue"):
            return {"status": "ok", "stdout": "1|me|RUNNING|1:00|2:00|1|n1|a\n"
                                              "2|me|PENDING|0:00|2:00|1|(Prio)|b\n"}
        return {"status": "ok", "stdout": "gpu|up|1-00|4|idle|32|64000\n"
                                          "gpu|up|1-00|6|alloc|32|64000\n"}

    with mock.patch.object(hpc.cli, "run_bash", _bash), \
            mock.patch.object(slurm_site, "_HAS_SLURM", True), \
            mock.patch.object(hpc, "_recorded_jobs", return_value={}), \
            mock.patch.object(hpc, "_host_tree", return_value=[]):
        out = _report(hpc)
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
