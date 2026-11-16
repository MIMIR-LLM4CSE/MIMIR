"""A workspace's own plugins and servers run once accepted, and again only unchanged.

A cloned repository's ``.mimir/plugins/x.py`` is imported into the client the moment it
starts. In VS Code the folder's trust answers for it; at a terminal, ``trust`` asks.
"""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from unittest import mock

from mimir.client.extensions import trust


class WorkspaceCodeTrustTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._state = os.path.join(self._tmp.name, "global")
        patches = [
            mock.patch.object(trust, "GLOBAL_STATE_DIR", self._state),
            mock.patch.dict(trust._decided, {}, clear=True),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(trust.TRUSTED_ENV, None)
        self.plugin = os.path.join(self._tmp.name, "x.py")
        with open(self.plugin, "w") as fh:
            fh.write("print('planted')\n")
        self.asked: list[list[str]] = []

    def _ask(self, answer: bool):
        def ask(paths: list[str]) -> bool:
            self.asked.append(paths)
            return answer
        return ask

    def test_a_planted_plugin_is_not_loaded_unasked(self) -> None:
        self.assertEqual(trust.trusted_only([self.plugin], self._ask(False)), [])
        self.assertEqual(self.asked, [[os.path.realpath(self.plugin)]])

    def test_accepted_once_it_loads_without_asking_again(self) -> None:
        self.assertEqual(trust.trusted_only([self.plugin], self._ask(True)), [self.plugin])
        trust._decided.clear()   # a new process
        self.assertEqual(trust.trusted_only([self.plugin], self._ask(False)), [self.plugin])
        self.assertEqual(len(self.asked), 1)
        store = os.path.join(self._state, "trusted_extensions.json")
        self.assertEqual(stat.S_IMODE(os.stat(store).st_mode) & 0o077, 0)

    def test_a_changed_file_is_asked_about_again(self) -> None:
        trust.trusted_only([self.plugin], self._ask(True))
        trust._decided.clear()
        with open(self.plugin, "a") as fh:
            fh.write("import os; os.system('id')\n")
        self.assertEqual(trust.trusted_only([self.plugin], self._ask(False)), [])
        self.assertEqual(len(self.asked), 2)

    def test_nobody_at_the_terminal_loads_nothing_new(self) -> None:
        with mock.patch.object(trust.sys, "stdin", None):
            self.assertEqual(trust.trusted_only([self.plugin]), [])

    def test_a_folder_vscode_trusts_asks_nothing(self) -> None:
        os.environ[trust.TRUSTED_ENV] = "1"
        self.assertEqual(trust.trusted_only([self.plugin], self._ask(False)), [self.plugin])
        self.assertEqual(self.asked, [])

    def test_the_loaders_go_through_it(self) -> None:
        from mimir.client.extensions.plugins import _import_packs
        from mimir.client.extensions.servers import discover_user_servers
        os.rename(self.plugin, os.path.join(self._tmp.name, "server_x.py"))
        with mock.patch.object(trust, "_ask_terminal", self._ask(False)):
            self.assertEqual(_import_packs(self._tmp.name).packs, [])
            self.assertEqual(discover_user_servers(self._tmp.name), {})


if __name__ == "__main__":
    unittest.main()
