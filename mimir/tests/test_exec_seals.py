"""The seal contract on its own: per owner, per conversation, and an unreadable seal is no
opinion."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from mimir.shared import exec_seals


class ExecSealsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        env = mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": os.path.join(tmp.name, "state"),
                                           "MIMIR_SESSION_ID": "s1"})
        env.start()
        self.addCleanup(env.stop)

    def _seal(self, owner: str, *names: str, **kw) -> list[str]:
        paths = [os.path.join(self.dir, n) for n in names]
        exec_seals.seal(owner, paths, refusal=f"{owner} refuses", hint=f"use {owner}", **kw)
        return paths

    def test_two_owners_seal_side_by_side(self) -> None:
        a = self._seal("bench", "a.py")
        b = self._seal("harness", "b.bin")
        sealed = exec_seals.sealed()
        self.assertEqual(sealed[os.path.realpath(a[0])]["refusal"], "bench refuses")
        self.assertEqual(sealed[os.path.realpath(b[0])]["hint"], "use harness")
        exec_seals.unseal("bench")
        self.assertEqual(list(exec_seals.sealed()), [os.path.realpath(b[0])])

    def test_a_seal_belongs_to_its_conversation(self) -> None:
        self._seal("bench", "a.py")
        self.assertEqual(exec_seals.sealed("s2"), {})
        self.assertEqual(len(exec_seals.sealed("s1")), 1)

    def test_the_measured_files_are_named_by_the_seal(self) -> None:
        source = self._seal("bench", "a.py", "a.bin", measured=[os.path.join(self.dir, "a.py")])[0]
        self.assertEqual(exec_seals.measured_sources(), {source})

    def test_an_unreadable_seal_gives_no_opinion(self) -> None:
        good = self._seal("bench", "a.py")
        self._seal("broken", "b.py")
        path = os.path.join(os.environ["MIMIR_STATE_DIR"], "sessions", "s1",
                            exec_seals.SEALS_DIRNAME, "broken.json")
        with open(path, "w") as fh:
            fh.write("{not json")
        self.assertEqual(list(exec_seals.sealed()), [os.path.realpath(good[0])])

    def test_unsealing_nothing_is_fine(self) -> None:
        exec_seals.unseal("never-sealed")
        self.assertEqual(exec_seals.sealed(), {})


if __name__ == "__main__":
    unittest.main()
