"""Read-only servers may read trusted out-of-workspace locations (proxy/HPC caches).

Reading agent-produced run artefacts under ~/.cache/proxy_bench etc. was a
recurring dead-end ("Path outside workspace"). The read/search server now admits
those as extra read roots, while the write server (server_files) stays strict and
arbitrary paths remain blocked.

Run:
    python -m unittest mimir.tests.test_read_roots -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "servers" / "_shared"
_WORKSPACE = Path(__file__).resolve().parents[1] / "servers" / "workspace"
for _p in (_SHARED, _WORKSPACE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import root_paths  # noqa: E402


class ResolveExtraRootsTests(unittest.TestCase):
    def test_extra_root_admits_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as extra:
            target = os.path.join(extra, "sub", "run.log")
            os.makedirs(os.path.dirname(target))
            open(target, "w").close()
            out = root_paths.resolve_path_in_root(
                target, root, "file root", extra_roots=[extra])
            self.assertEqual(out, os.path.realpath(target))

    def test_without_extra_root_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as extra:
            target = os.path.join(extra, "run.log")
            open(target, "w").close()
            with self.assertRaises(ValueError):
                root_paths.resolve_path_in_root(target, root, "file root")

    def test_unrelated_path_blocked_even_with_extra_roots(self) -> None:
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as extra:
            with self.assertRaises(ValueError):
                root_paths.resolve_path_in_root(
                    "/etc/passwd", root, "file root", extra_roots=[extra])

    def test_relative_paths_still_resolve_against_primary_root_only(self) -> None:
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as extra:
            open(os.path.join(root, "a.py"), "w").close()
            out = root_paths.resolve_path_in_root(
                "a.py", root, "file root", extra_roots=[extra])
            self.assertEqual(out, os.path.realpath(os.path.join(root, "a.py")))


class SearchServerReadRootsTests(unittest.TestCase):
    def test_proxy_cache_is_a_read_root(self) -> None:
        import server_search as ss
        roots = ss._extra_read_roots()
        self.assertIn(os.path.expanduser("~/.cache/proxy_bench"), roots)

    def test_env_extends_read_roots(self) -> None:
        import server_search as ss
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("MIMIR_EXTRA_FILE_ROOTS")
            os.environ["MIMIR_EXTRA_FILE_ROOTS"] = d
            try:
                self.assertIn(d, ss._extra_read_roots())
            finally:
                if old is None:
                    os.environ.pop("MIMIR_EXTRA_FILE_ROOTS", None)
                else:
                    os.environ["MIMIR_EXTRA_FILE_ROOTS"] = old

    def test_write_server_stays_strict(self) -> None:
        # server_files._safe must NOT admit the proxy cache (writes stay in-workspace).
        import server_files as sf
        with self.assertRaises(ValueError):
            sf._safe(os.path.expanduser("~/.cache/proxy_bench/x.log"))


if __name__ == "__main__":
    unittest.main()
