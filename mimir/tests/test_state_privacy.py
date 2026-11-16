"""The state directories are private: on a login node, a home others can traverse would
otherwise let them read every transcript, plan and job record MIMIR writes."""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from unittest import mock

from mimir.shared.state_paths import ensure_private_dir, make_state_private


class StatePrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_an_open_state_directory_is_closed(self) -> None:
        home = os.path.join(self._tmp.name, "state")
        os.makedirs(home, mode=0o775)
        os.chmod(home, 0o775)
        new = os.path.join(home, "ws-1234")
        self.assertEqual(make_state_private(home, new), [])
        for d in (home, new):
            self.assertEqual(stat.S_IMODE(os.stat(d).st_mode), 0o700, d)

    def test_the_home_directory_and_its_ancestors_are_refused(self) -> None:
        fake_home = os.path.join(self._tmp.name, "u")
        os.makedirs(fake_home)
        with mock.patch.dict(os.environ, {"HOME": fake_home}):
            self.assertFalse(ensure_private_dir(fake_home))
            self.assertFalse(ensure_private_dir(self._tmp.name))
            self.assertEqual(make_state_private(fake_home), [fake_home])
            self.assertTrue(ensure_private_dir(os.path.join(fake_home, ".mimir")))


if __name__ == "__main__":
    unittest.main()
