"""The mandatory check, now performed in-process.

Two things are being defended here, and the second is the harder one:

- the floor *catches* the file written half-way, in every language, with nothing
  installed;
- the floor *does not* accuse code that is fine. A false positive charges a repair
  budget against a correct file and sends the model rewriting it, so every language
  case below carries its own "this is legal, leave it alone" counterpart, and
  ``test_the_repo_checks_clean`` runs the whole thing over this repository.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

from mimir.client.guardrails.builtin_check import check_file, sweep_builtin_checks


def _write(directory: str, name: str, content) -> str:
    path = os.path.join(directory, name)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as handle:
        handle.write(content)
    return path


class BuiltinCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()

    def _status(self, name: str, content) -> str:
        return check_file(_write(self.dir, name, content)).status

    def test_a_stdlib_parser_answers_where_there_is_one(self) -> None:
        for name, content in (
            ("a.py", "def f():\n    return 1\n"),
            ("a.json", '{"a": [1, 2]}'),
            ("a.toml", 'x = 1\n'),
        ):
            with self.subTest(file=name):
                outcome = check_file(_write(self.dir, name, content))
                self.assertEqual(outcome.status, "ok")
                # `.toml` degrades to the structural scan below Python 3.11, which is
                # the point of the fallback: the answer is never "not checked".
                self.assertIn(outcome.tier, ("syntax", "structural"))

    def test_xml_is_parsed_for_real(self) -> None:
        self.assertEqual(self._status("a.xml", "<r><c a='1'/></r>"), "ok")
        self.assertEqual(
            self._status("a.svg", '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'),
            "ok",
        )
        outcome = check_file(_write(self.dir, "b.xml", "<r><c></r>"))
        self.assertEqual(outcome.status, "fail")
        self.assertIn("line 1", outcome.detail)

    def test_an_entity_declaration_declines_rather_than_expands(self) -> None:
        # The stdlib parser expands internal entities, so a billion-laughs file would
        # be a denial of service performed by the checker itself. Declining hands it to
        # the structural scan, which is the floor's answer to every uncertain reading.
        bomb = (
            '<!DOCTYPE l [<!ENTITY a "aa"><!ENTITY b "&a;&a;">]>\n<l>&b;</l>'
        )
        outcome = check_file(_write(self.dir, "bomb.xml", bomb))
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.checker, "structural")

    def test_ini_is_parsed_for_real(self) -> None:
        self.assertEqual(self._status("a.ini", "[s]\nk = v\n"), "ok")
        # `%(name)s` is legal INI and only interpolation would object, which is why the
        # raw parser is used: a logging config must not be reported as broken.
        self.assertEqual(
            self._status("log.ini", "[formatters]\nfmt = %(asctime)s %(message)s\n"), "ok",
        )
        duplicate = check_file(_write(self.dir, "d.ini", "[s]\nk = v\nk = w\n"))
        self.assertEqual(duplicate.status, "fail")
        self.assertIn("line 3", duplicate.detail)
        malformed = check_file(_write(self.dir, "m.ini", "[s]\nkey without equals\n"))
        self.assertEqual(malformed.status, "fail")
        self.assertIn("line 2", malformed.detail)

    def test_a_cfg_that_is_not_ini_is_not_accused_of_being_bad_ini(self) -> None:
        # `.cfg` is claimed by tools that put something else in it; configparser's first
        # complaint would be about the wrong grammar entirely.
        outcome = check_file(_write(self.dir, "x.cfg", "not ini at all\njust prose (\n"))
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.checker, "structural")

    def test_every_extension_with_a_real_parser_is_tracked_as_produced_work(self) -> None:
        # A parser nothing routes a file to is dead code: the sweep only ever sees
        # extensions `SOURCE_FILE_EXTENSIONS` records as modified.
        from mimir.client.guardrails.builtin_check import _PARSER_BY_EXTENSION
        from mimir.client.context.signals import SOURCE_FILE_EXTENSIONS
        # `.svg` and the project-file dialects are assets, not produced work; the rest
        # must be reachable.
        assets = {".svg", ".vcxproj", ".csproj", ".ui", ".rng", ".wsdl", ".pyw"}
        self.assertEqual(
            set(_PARSER_BY_EXTENSION) - assets - set(SOURCE_FILE_EXTENSIONS), set(),
        )

    def test_a_parser_reports_the_offending_line(self) -> None:
        outcome = check_file(_write(self.dir, "a.py", "x = 1\ndef f(:\n"))
        self.assertEqual(outcome.status, "fail")
        self.assertEqual(outcome.tier, "syntax")
        self.assertIn("line 2", outcome.detail)

    def test_truncation_is_caught_in_every_language(self) -> None:
        for name, content in (
            ("a.c", "int main(){\n  if (a) {\n"),
            ("a.f90", "program p\n  x = (a + b\nend program\n"),
            ("a.rs", 'fn main() { println!("hi");\n'),
            ("a.ts", "function f() {\n"),
            ("a.java", "class A {\n"),
            ("a.sh", "if [ -f x ]; then\n  echo hi\n"),
            ("a.json", "{"),
            ("a.py", "def f(\n"),
        ):
            with self.subTest(file=name):
                self.assertEqual(self._status(name, content), "fail")

    def test_legal_code_is_left_alone(self) -> None:
        # Each of these once read as unbalanced to a naive scan: an angle bracket, a
        # `case` pattern, a regular-expression character class, a heredoc body.
        for name, content in (
            ("a.c", "#include <stdio.h>\nint m(){ /* it's fine */ return a < b; }\n"),
            ("a.f90", "program p\n  ! it's a comment\n  x = (a + b)\nend program\n"),
            ("a.sh", "case $a in\n  b) echo hi;;\nesac\n"),
            ("a.sh", "cat <<EOF\nif this were code it would never close\nEOF\necho ok\n"),
            ("a.ts", "const r = /[a-z)]/g;\nconst t = `a ${b} c`;\n"),
            ("a.js", "const x = a / b / c;\nfunction f() { return 1; }\n"),
            ("a.md", "Its (unbalanced parenthesis and an apostrophe\n"),
        ):
            with self.subTest(file=name, content=content[:20]):
                self.assertEqual(self._status(name, content), "ok")

    def test_a_leftover_conflict_marker_is_a_defect(self) -> None:
        outcome = check_file(_write(self.dir, "a.c", "<<<<<<< HEAD\nint a;\n"))
        self.assertEqual(outcome.status, "fail")
        self.assertIn("conflict marker", outcome.detail)

    def test_an_unclosed_block_comment_is_a_defect(self) -> None:
        self.assertEqual(self._status("a.c", "/* never closed\nint a;\n"), "fail")

    def test_what_is_not_text_is_unreadable_rather_than_wrong(self) -> None:
        self.assertEqual(self._status("a.py", b"\x00\x01\x02"), "unreadable")
        self.assertEqual(self._status("b.py", b"\xff\xfe\x00"), "unreadable")

    def test_an_unknown_extension_gets_the_prose_floor(self) -> None:
        # No grammar is assumed for a file nothing here recognises, so only what is
        # wrong in any text file is looked for.
        self.assertEqual(self._status("notes.zzz", "a ( b [ c\n"), "ok")
        self.assertEqual(self._status("notes.zzz", "<<<<<<< HEAD\n"), "fail")

    def test_the_repo_checks_clean(self) -> None:
        # The false-positive guard that matters: every file this project tracks,
        # minified vendor bundles included.
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        listing = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True,
        )
        if listing.returncode != 0:
            self.skipTest("not a git checkout")
        rejected = []
        for relative in listing.stdout.split():
            path = os.path.join(root, relative)
            if not os.path.isfile(path):
                continue
            outcome = check_file(path)
            if outcome.status == "fail":
                rejected.append(f"{relative}: {outcome.detail}")
        self.assertEqual(rejected, [])


class SweepTests(unittest.TestCase):
    """One pass over the final content, not one per edit."""

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()

    def _context(self, *paths: str) -> dict:
        return {
            "code_mutation_started": True,
            "dirty_written_files": set(paths),
            "validated_files": set(),
            "unverifiable_files": set(),
            "validation_tier_by_file": {},
            "validation_fail_count_by_file": {},
            "workflow_state": "validate",
            "nudge_counts": {},
        }

    def _resolver(self):
        return lambda path: os.path.join(self.dir, path)

    def test_a_clean_file_is_credited_structurally(self) -> None:
        _write(self.dir, "engine.rs", "fn main() {}\n")
        ec = self._context("engine.rs")
        sweep_builtin_checks(ec, self._resolver())
        self.assertEqual(ec["validated_files"], {"engine.rs"})
        self.assertEqual(ec["validation_tier_by_file"]["engine.rs"], "structural")

    def test_a_rejected_file_charges_the_retry_budget_and_keeps_the_diagnostic(self) -> None:
        _write(self.dir, "solver.py", "def f(:\n")
        ec = self._context("solver.py")
        sweep_builtin_checks(ec, self._resolver())
        self.assertEqual(ec["validated_files"], set())
        self.assertEqual(ec["validation_fail_count_by_file"]["solver.py"], 1)
        self.assertIn("line 1", ec["builtin_check_findings"]["solver.py"])
        self.assertEqual(ec["workflow_state"], "edit")

    def test_an_unchanged_file_is_not_re_checked(self) -> None:
        path = _write(self.dir, "solver.py", "def f(:\n")
        ec = self._context("solver.py")
        sweep_builtin_checks(ec, self._resolver())
        sweep_builtin_checks(ec, self._resolver())
        sweep_builtin_checks(ec, self._resolver())
        # Three calls, one charge: the gate is consulted from several places and the
        # budget must not drain just because the loop asked more than once.
        self.assertEqual(ec["validation_fail_count_by_file"]["solver.py"], 1)

        _write(self.dir, "solver.py", "def g(:\n")
        os.utime(path, (0, 0))
        sweep_builtin_checks(ec, self._resolver())
        self.assertEqual(ec["validation_fail_count_by_file"]["solver.py"], 2)

    def test_a_repair_clears_the_finding(self) -> None:
        _write(self.dir, "solver.py", "def f(:\n")
        ec = self._context("solver.py")
        sweep_builtin_checks(ec, self._resolver())
        path = _write(self.dir, "solver.py", "def f():\n    return 1\n")
        os.utime(path, (0, 0))
        sweep_builtin_checks(ec, self._resolver())
        self.assertEqual(ec["builtin_check_findings"], {})
        self.assertEqual(ec["validated_files"], {"solver.py"})


if __name__ == "__main__":
    unittest.main()
