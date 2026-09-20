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


class GithubFileWindowTests(unittest.TestCase):
    """A GitHub file is read in pages, with the keys the local read established.

    Two losses this replaced. A file over 256 KB was refused outright — a total loss
    of the file to avoid a large one — and a file under it came back whole, up to
    ~65k tokens, with no way to ask for part of it. Same key names as
    ``read_file_lines`` on purpose: the client's ``_build_continuation_hint`` already
    reads them, so a paged GitHub read inherits that for free.
    """

    _TEXT = "".join(f"line {i}\n" for i in range(1, 1001))

    def test_the_first_page_stops_at_the_cap_and_says_where_to_resume(self):
        w = server_github._line_window(self._TEXT, 1, 0)
        self.assertEqual(w["start_line"], 1)
        self.assertEqual(w["end_line"], server_github._MAX_READ_LINES)
        self.assertEqual(w["total_lines"], 1000)
        self.assertTrue(w["truncated"])
        self.assertEqual(w["next_start_line"], server_github._MAX_READ_LINES + 1)
        self.assertTrue(w["content"].startswith("line 1\n"))

    def test_resuming_walks_the_file_without_gap_or_overlap(self):
        seen, start, pages = [], 1, 0
        while start and pages < 10:
            w = server_github._line_window(self._TEXT, start, 0)
            seen.append(w["content"])
            start = w.get("next_start_line")
            pages += 1
        self.assertEqual("".join(seen), self._TEXT)

    def test_an_explicit_range_is_honoured_and_still_capped(self):
        w = server_github._line_window(self._TEXT, 200, 260)
        self.assertEqual((w["start_line"], w["end_line"]), (200, 260))
        self.assertEqual(w["lines_returned"], 61)
        self.assertNotIn("line_cap", w)          # the range was under the cap
        wide = server_github._line_window(self._TEXT, 1, 999)
        self.assertEqual(wide["line_cap"], server_github._MAX_READ_LINES)

    def test_the_last_page_does_not_claim_a_continuation(self):
        w = server_github._line_window(self._TEXT, 801, 0)
        self.assertEqual(w["end_line"], 1000)
        self.assertNotIn("truncated", w)
        self.assertNotIn("next_start_line", w)

    def test_a_start_past_the_end_is_empty_rather_than_an_error(self):
        w = server_github._line_window(self._TEXT, 5000, 0)
        self.assertEqual(w["content"], "")
        self.assertEqual(w["lines_returned"], 0)
        self.assertEqual(w["total_lines"], 1000)

    def test_an_empty_file_is_not_a_special_case(self):
        w = server_github._line_window("", 1, 0)
        self.assertEqual(w["total_lines"], 0)
        self.assertEqual(w["content"], "")
