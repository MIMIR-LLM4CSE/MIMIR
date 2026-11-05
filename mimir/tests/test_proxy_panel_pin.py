"""The Optimisation section shows the proxy the reader pinned, else the agent's active
one; no run moves it."""

import os
import tempfile
import unittest
from unittest import mock

from mimir.servers.proxy._lib import layout
from mimir.servers.proxy._lib.procs import new_run_dir, update_opt_active_link
from mimir.servers.proxy._ops import panel
from mimir.shared.atomic import write_json, write_text


class _Store(unittest.TestCase):
    """A proxy_bench of its own, two proxies initialised in it, this conversation 'conv'."""

    NAMES = ("a", "b")

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        saved = layout._ROOT
        layout._ROOT = os.path.join(tmp.name, "proxy_bench")
        self.addCleanup(setattr, layout, "_ROOT", saved)
        env = mock.patch.dict(os.environ, {"MIMIR_SESSION_ID": "conv"})
        env.start()
        self.addCleanup(env.stop)
        for name in self.NAMES:
            write_json(layout.opt_config_file(name), {"proxy_name": name, "benchmark_name": "bench"})

    def point(self, name: str, at: float) -> None:
        """This conversation's pointer on *name*, set at *at* (init, a re-baseline)."""
        path = layout.active_session_file()
        write_text(path, name)
        os.utime(path, (at, at))

    def launch(self, name: str, at: float, by: str = "conv") -> None:
        """A run of *name* started at *at* by session *by*."""
        with mock.patch.dict(os.environ, {"MIMIR_SESSION_ID": by}):
            run_dir = new_run_dir(layout.opt_session_runs_dir(name))
        with open(os.path.join(run_dir, "start_time"), "w") as fh:
            fh.write(str(at))
        update_opt_active_link(name, run_dir)

    def shown(self, picked: str = "") -> dict:
        return panel.report(picked)


class PinTests(_Store):
    def test_without_a_pin_it_shows_the_agent_s_active_proxy(self) -> None:
        self.point("a", 1000)
        out = self.shown()
        self.assertEqual(out["choice"]["options"][0]["label"], "the agent's active proxy · a")
        self.assertEqual(out["choice"]["value"], "")

    def test_runs_launched_on_other_proxies_do_not_move_it(self) -> None:
        """Neither the agent's nor its sub-agents': the section changes when the reader
        pins another proxy, or when the agent initialises one."""
        self.point("a", 1000)
        self.launch("b", 1200)
        self.launch("b", 1300, by="conv/subagents/c1")
        self.assertEqual(self.shown()["choice"]["options"][0]["label"], "the agent's active proxy · a")

    def test_a_conversation_with_no_active_proxy_asks_for_a_pin(self) -> None:
        self.launch("b", 1200, by="other")
        out = self.shown()
        self.assertEqual(out["detail"], "Pick the proxy to follow.")
        self.assertIn("choice", out)

    def test_a_pinned_proxy_stays_whatever_runs(self) -> None:
        self.point("a", 1000)
        self.launch("a", 1200)
        out = self.shown("b")
        self.assertEqual(out["choice"]["value"], "b")
        self.assertIn("state", [line["label"] for line in out["lines"]])


class ChoiceTests(_Store):
    def test_the_reader_s_pick_is_shown_and_echoed(self) -> None:
        self.point("a", 1000)
        out = self.shown("b")
        self.assertEqual(out["choice"]["value"], "b")
        self.assertEqual([o["value"] for o in out["choice"]["options"]], ["", "a", "b"])
        self.assertNotIn("proxy", [line["label"] for line in out["lines"]])

    def test_a_pick_that_is_no_session_falls_back_to_the_agent_s(self) -> None:
        self.point("a", 1000)
        self.assertEqual(self.shown("gone")["choice"]["value"], "")


class OneSessionTests(_Store):
    NAMES = ("a",)

    def test_one_session_offers_no_choice_and_names_it(self) -> None:
        self.point("a", 1000)
        out = self.shown()
        self.assertNotIn("choice", out)
        self.assertEqual(out["lines"][0], {"label": "proxy", "value": "a"})


if __name__ == "__main__":
    unittest.main()
