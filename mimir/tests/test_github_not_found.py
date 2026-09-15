"""A 404 from ``github_get_file`` reports what the repository actually holds.

A guessed file name is the common case behind a 404, and a bare "Not Found" leaves
the model with nothing to correct: in one FUnTiDES session it retried the same
missing path five times across two tools. The error now walks up to the deepest
directory that does exist, names its entries, and points at the closest names, so
the next call is informed rather than another guess. Every request here is faked —
the test must never reach GitHub.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "external"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location(
    "server_github", SERVERS_DIR / "external" / "server_github.py"
)
server_github = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server_github)

from mimir.servers._shared.responses import err, ok  # noqa: E402


def _entries(*names: str) -> list[dict]:
    return [{"name": name, "type": "file"} for name in names]


class _FakeGitHub:
    """Stands in for ``_request``: serves a fixed contents tree, 404s elsewhere."""

    def __init__(self, tree: dict[str, object]):
        self.tree = tree
        self.calls: list[str] = []

    def __call__(self, path: str, params: dict | None = None) -> dict:
        self.calls.append(path)
        prefix = "/repos/o/r/contents/"
        key = path[len(prefix):] if path.startswith(prefix) else path
        if key in self.tree:
            return ok({"data": self.tree[key]})
        return err("Not Found", hint="Check repository owner/name.", http_status=404)


class GithubGetFileNotFoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_request = server_github._request

    def tearDown(self) -> None:
        server_github._request = self._real_request

    def _install(self, tree: dict[str, object]) -> _FakeGitHub:
        fake = _FakeGitHub(tree)
        server_github._request = fake
        return fake

    def test_missing_file_reports_sibling_listing_and_near_matches(self) -> None:
        self._install({
            "src/specfem2D": _entries(
                "compute_forces_viscoacoustic.f90", "compute_energy.f90", "Makefile"
            ),
        })
        result = server_github.github_get_file(
            "o", "r", "src/specfem2D/compute_forces_acoustic.f90"
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["http_status"], 404)
        self.assertEqual(result["listed_path"], "src/specfem2D")
        self.assertIn("compute_forces_viscoacoustic.f90", result["entries"])
        # The correct name is the first thing offered, not buried in the listing.
        self.assertEqual(result["near_matches"][0], "compute_forces_viscoacoustic.f90")
        self.assertTrue(result["hint"].startswith("Closest names to"))

    def test_walks_up_to_the_deepest_directory_that_exists(self) -> None:
        fake = self._install({"src": _entries("specfem2D", "shared")})
        result = server_github.github_get_file("o", "r", "src/nope/deeper/file.f90")
        self.assertEqual(result["listed_path"], "src")
        self.assertEqual(result["entries"], ["shared", "specfem2D"])
        self.assertEqual(result["near_matches"], [])
        # One failed fetch, then one listing attempt per level up to 'src'.
        self.assertEqual(len(fake.calls), 4)

    def test_root_level_miss_lists_the_repository_root(self) -> None:
        self._install({"": _entries("README.md", "LICENSE")})
        result = server_github.github_get_file("o", "r", "READMEE.md")
        self.assertEqual(result["listed_path"], "")
        self.assertIn("the repository root exists and contains", result["hint"])
        self.assertEqual(result["near_matches"], ["README.md"])

    def test_listing_is_capped_and_says_how_many_are_hidden(self) -> None:
        many = [f"file{i:03d}.f90" for i in range(server_github._MAX_SIBLINGS + 7)]
        self._install({"src": _entries(*many)})
        result = server_github.github_get_file("o", "r", "src/absent_name.txt")
        self.assertEqual(len(result["entries"]), server_github._MAX_SIBLINGS)
        self.assertIn("(7 more not shown.)", result["hint"])

    def test_unlistable_repository_keeps_the_original_error(self) -> None:
        fake = self._install({})
        result = server_github.github_get_file("o", "r", "src/specfem2D/x.f90")
        self.assertEqual(result["error"], "Not Found")
        self.assertEqual(result["hint"], "Check repository owner/name.")
        self.assertNotIn("entries", result)
        # The failed fetch, then one attempt per level up to and including the root.
        self.assertEqual(len(fake.calls), 4)

    def test_non_404_errors_are_passed_through_untouched(self) -> None:
        calls: list[str] = []

        def rate_limited(path: str, params: dict | None = None) -> dict:
            calls.append(path)
            return err("API rate limit exceeded", http_status=403)

        server_github._request = rate_limited
        result = server_github.github_get_file("o", "r", "src/specfem2D/x.f90")
        self.assertEqual(result["error"], "API rate limit exceeded")
        self.assertNotIn("entries", result)
        # No listing walk: a rate limit would only burn more of the same budget.
        self.assertEqual(len(calls), 1)

    def test_successful_fetch_costs_no_extra_request(self) -> None:
        import base64

        fake = self._install({
            "src/a.f90": {
                "type": "file", "path": "src/a.f90", "size": 5, "sha": "abc",
                "encoding": "base64", "content": base64.b64encode(b"hello").decode(),
            },
        })
        result = server_github.github_get_file("o", "r", "src/a.f90")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], "hello")
        self.assertEqual(len(fake.calls), 1)

    def test_ref_is_named_in_the_error_and_carried_into_the_listing(self) -> None:
        seen: list[dict | None] = []
        tree = {"src": _entries("a.f90")}

        def tracking(path: str, params: dict | None = None) -> dict:
            seen.append(params)
            key = path[len("/repos/o/r/contents/"):]
            if key in tree:
                return ok({"data": tree[key]})
            return err("Not Found", http_status=404)

        server_github._request = tracking
        result = server_github.github_get_file("o", "r", "src/b.f90", ref="devel")
        self.assertIn("at ref 'devel'", result["error"])
        self.assertEqual(seen, [{"ref": "devel"}, {"ref": "devel"}])


if __name__ == "__main__":
    unittest.main()
