"""Coverage measurement for the bash → blackboard pipeline.

``test_bash_classify.py`` tests the classifier *case by case*: it pins that the forms
someone already thought of map to the right ``Kind``. What it cannot say is which
fraction of the shell a model actually writes ends up crediting anything — and that
is the number that matters, because the shell is an unbounded language projected into
a small vocabulary of states. Every form the classifier does not recognise fails
**silently**: the command runs, the file is edited or validated, and the blackboard
never hears about it. Worse, the failure grows with model capability — a stronger
model writes more varied shell, so the blind spots widen exactly as the agent improves.

So this module measures instead of enumerating, on the pattern of
``test_server_contracts.test_every_refused_path_is_one_the_user_can_be_asked_about``:

1. :data:`_CREDITING_CORPUS` — realistic commands, each annotated with what it must
   credit. Assertions run the **whole** pipeline (``record_tool_observation``), not
   just ``classify_bash_command``, because the gap between "the segment has the right
   kind" and "the execution context learned something" is where regressions live.
2. :data:`_KNOWN_UNCREDITED` — the frozen blind spots, asserted to credit *nothing*.
   Making one of them work must delete its entry consciously; introducing a *new*
   blind spot breaks (1). The list is the executable documentation of the blind surface.
3. A floor on the credit rate, so a whole unhandled command family shows up as a
   number rather than dissolving into the suite.

Pure-Python + stubs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import json
import os
import types
import unittest

from mimir.client.context import SOURCE_FILE_EXTENSIONS
from mimir.client.context.capabilities import ToolCaps
from mimir.client.context.execution_context import build_execution_context
from mimir.client.guardrails.observations import record_tool_observation
from mimir.client.guardrails.policy.bash_classify import classify_bash_command

BASH_TOOL = "bash_run"


def _stub_agent():
    """An agent stub carrying only what the observation layer reads.

    ``tool_caps`` must declare the ``command_prefix`` scope kind: that declaration is
    how ``_carries_shell_command`` recognises a shell tool without naming it, so a stub
    without it silently observes nothing and every assertion here would pass vacuously.
    """
    caps = {
        BASH_TOOL: ToolCaps(
            name=BASH_TOOL,
            capabilities=frozenset(),
            scope={"kind": "command_prefix", "args": ["command"]},
        )
    }
    return types.SimpleNamespace(
        tool_caps=caps,
        _parse_tool_payload=lambda result: json.loads(result) if result else {},
        _normalize_workspace_path=lambda p: os.path.normpath(p) if p else "",
        _is_code_filepath=lambda p: os.path.splitext(p)[1].lower() in SOURCE_FILE_EXTENSIONS,
    )


def _run(command: str, *, status: str = "ok", stdout: str = "out",
         dirty: tuple[str, ...] = ()) -> dict:
    """Drive the full observation pipeline for one bash call; return the context.

    *dirty* pre-seeds ``dirty_written_files`` because validation credit is only ever
    given to a file the model has already written — without it a ``pytest solver.py``
    assertion would test nothing.
    """
    ec = build_execution_context()
    for path in dirty:
        ec["dirty_written_files"].add(path)
        ec["code_mutation_started"] = True
    payload = json.dumps({"status": status, "stdout": stdout})
    record_tool_observation(_stub_agent(), BASH_TOOL, {"command": command}, payload, ec)
    return ec


# ── What a credited command must have taught the blackboard ───────────────────
#
# Each entry is (command, expectation). The expectation keys map to the fields
# ``_observe_command`` / ``_observe_bash_validation`` write, and only the keys present
# are asserted — an entry states what the command *must* establish, not everything it
# happens to touch.
#
#   read      -> paths that must land in read_files
#   search    -> the `searched` flag must be set
#   inspect   -> paths that must land in inspected_dirs
#   write     -> paths that must land in dirty_written_files
#   validate  -> paths that must land in validated_files (implies they were dirty)
#   judge     -> paths a run left awaiting the model's verdict. Exit 0 from something
#                that *executes* proves the program ended, not that its answer is
#                right, so those commands park here instead of crediting validation;
#                a checker (py_compile/ruff/mypy/a compiler) still credits directly.
#   project   -> a green whole-project validator: clears every pending file
#   tests_run -> paths that must land in tests_run (feeds the regression nudge)
#   env       -> an environment mutation must be recorded
#   action    -> action_op_count must have advanced (a substantive operation ran)
_CREDITING_CORPUS: list[tuple[str, dict]] = [
    # ── discovery ────────────────────────────────────────────────────────────
    ("cat solver.py", {"read": ["solver.py"]}),
    ("head -50 src/mesh.c", {"read": ["src/mesh.c"]}),
    ("sed -n '1,40p' src/mesh.c", {"read": ["src/mesh.c"]}),
    ("grep -rn 'assemble' src", {"search": True}),
    ("rg --hidden 'TODO' .", {"search": True}),
    ("ls -la src", {"inspect": ["src"]}),
    ("find . -name '*.f90'", {"inspect": ["."]}),
    ("cd src && cat mesh.c", {"read": ["src/mesh.c"]}),

    # ── mutation through the shell ───────────────────────────────────────────
    ("sed -i 's/foo/bar/' solver.py", {"write": ["solver.py"], "action": True}),
    ("cp solver.py backup.py", {"write": ["backup.py"], "action": True}),
    ("mv old_solver.py solver.py", {"write": ["solver.py"], "action": True}),

    # ── validation, per file, across languages ───────────────────────────────
    ("python -m py_compile solver.py", {"validate": ["solver.py"]}),
    ("python -m pytest -q tests/test_solver.py",
     {"judge": ["tests/test_solver.py"], "tests_run": ["tests/test_solver.py"]}),
    ("pytest tests/test_solver.py",
     {"judge": ["tests/test_solver.py"], "tests_run": ["tests/test_solver.py"]}),
    ("ruff check solver.py", {"validate": ["solver.py"]}),
    ("mypy solver.py", {"validate": ["solver.py"]}),
    ("gcc -O2 -c src/mesh.c -o mesh.o", {"validate": ["src/mesh.c"]}),
    ("gfortran -O2 solver.f90 -o solver", {"validate": ["solver.f90"]}),
    ("nvcc -arch=sm_80 kernel.cu -o kernel", {"validate": ["kernel.cu"]}),
    # `node` runs a program as readily as it checks one, and the classifier keys on the
    # command head, so the pessimistic reading applies: ask for a verdict.
    ("node --check app.js", {"judge": ["app.js"]}),
    ("cd build && ctest", {"judge": ["pending.py"]}),
    # Running a program names the file it ran, but exit 0 says only that it ended.
    ("python solver.py", {"judge": ["solver.py"]}),

    # ── validation, whole project ────────────────────────────────────────────
    ("pytest", {"judge": ["pending.py"]}),
    ("pytest -q", {"judge": ["pending.py"]}),
    ("ruff check .", {"project": True}),
    ("mypy src/", {"project": True}),

    # ── chains the model actually writes ─────────────────────────────────────
    ("cd tests && pytest test_solver.py",
     {"judge": ["tests/test_solver.py"], "tests_run": ["tests/test_solver.py"]}),
    # The idiom the base prompt asks for — a one-off check inline rather than a file.
    # Parentheses alone make it unclassifiable, so it credits no file; but it plainly
    # ran, and what it printed is the whole point of running it.
    ('python -c "import solver; print(solver.residual())"', {"ran": True}),
    ("python -m py_compile solver.py && ruff check solver.py", {"validate": ["solver.py"]}),

    # ── environment ──────────────────────────────────────────────────────────
    ("pip install numpy", {"env": True}),
    ("conda create -n solve python=3.11", {"env": True}),
    ("module load cuda", {"env": True}),
]


# ── The frozen blind surface ──────────────────────────────────────────────────
#
# Forms a capable model writes that the pipeline credits NOTHING for. Each is a real
# hole, listed with the reason it exists — not an accident waiting to be discovered
# on a run. Two rules follow from the list being asserted:
#   * closing a hole means deleting its line here, deliberately, in the same change;
#   * opening a NEW one breaks the corpus assertions above, not this list.
# The reason matters when weighing a fix: each of these hides the command itself from
# the classifier, so not even its head can be read.
#
# Entries left this list when executions started owing a verdict: `make test`,
# `cmake --build build`, `./solver --check`, and every `python -c` form. They still
# credit no *validation* — a green `make clean` establishes nothing, and an inline
# payload names no file — but each is now recorded as a run whose output the model must
# account for, which is the one thing that can be said about them without knowing what
# they did.
_KNOWN_UNCREDITED: list[tuple[str, str]] = [
    ("tox -e py311", "unrecognised leading command — nothing in the head to read"),
    ("cat $(ls *.py | head -1)", "command substitution runs code, under a `cat` head"),
    ("bash -c 'pytest tests/'", "wrapper hides the real command behind a non-exec head"),
]


class CorpusCreditTests(unittest.TestCase):
    """Every corpus command must teach the blackboard what it claims to."""

    def _assert_credits(self, command: str, expect: dict) -> None:
        dirty = tuple(expect.get("validate", ())) or (
            ("pending.py",) if expect.get("project") or expect.get("ran") else ()
        )
        ec = _run(command, dirty=dirty)

        for path in expect.get("read", ()):
            self.assertIn(path, ec["read_files"], f"{command!r}: read not credited")
        if expect.get("search"):
            self.assertTrue(ec["searched"], f"{command!r}: search not credited")
        for path in expect.get("inspect", ()):
            self.assertIn(path, ec["inspected_dirs"], f"{command!r}: dir not credited")
        for path in expect.get("write", ()):
            self.assertIn(path, ec["dirty_written_files"], f"{command!r}: write not credited")
        for path in expect.get("validate", ()):
            self.assertIn(path, ec["validated_files"], f"{command!r}: check not credited")
        if expect.get("ran"):
            self.assertTrue(ec["runs"], f"{command!r}: run not recorded")
            self.assertFalse(ec["validated_files"],
                             f"{command!r}: an execution must not validate a file")
        if expect.get("project"):
            self.assertIn("pending.py", ec["validated_files"],
                          f"{command!r}: whole-project check did not clear pending files")
        for path in expect.get("tests_run", ()):
            self.assertIn(path, ec["tests_run"], f"{command!r}: test run not recorded")
        if expect.get("env"):
            self.assertTrue(ec.get("env_mutations"), f"{command!r}: env mutation not recorded")
        if expect.get("action"):
            self.assertGreater(ec["action_op_count"], 0, f"{command!r}: no action counted")

    def test_corpus_commands_credit_what_they_claim(self) -> None:
        for command, expect in _CREDITING_CORPUS:
            with self.subTest(command=command):
                self._assert_credits(command, expect)

    def test_a_failed_check_is_attributed_not_credited(self) -> None:
        """A red check must charge the file, not silently leave it unvalidated.

        The mirror of the corpus above: crediting on success is only half the
        contract, and a failure that lands nowhere lets a broken file reach the
        conclude gate looking merely un-checked.
        """
        ec = _run("ruff check tests/test_solver.py", status="error",
                  dirty=("tests/test_solver.py",))
        self.assertNotIn("tests/test_solver.py", ec["validated_files"])
        self.assertEqual(ec["validation_fail_count_by_file"].get("tests/test_solver.py"), 1)

    def test_a_failed_run_is_charged_to_the_run(self) -> None:
        ec = _run("pytest tests/test_solver.py", status="error",
                  dirty=("tests/test_solver.py",))
        self.assertEqual(ec["validation_fail_count_by_file"], {})
        self.assertEqual(ec["runs"]["pytest tests/test_solver.py"]["failures"], 1)


class BlindSurfaceTests(unittest.TestCase):
    """The known holes, frozen so closing one is a deliberate act."""

    def test_known_blind_spots_credit_nothing(self) -> None:
        for command, reason in _KNOWN_UNCREDITED:
            with self.subTest(command=command):
                self.assertFalse(
                    _credits_semantics(command),
                    f"{command!r} now credits something ({reason}) — if that is intended, "
                    "delete its entry from _KNOWN_UNCREDITED in the same change",
                )

    def test_the_frozen_list_is_exactly_the_blind_surface(self) -> None:
        """The list and the measurement must agree in *both* directions.

        The two lists are a claim about the pipeline; this is the claim checked against
        it. Drift either way is a silent lie: an entry that started crediting makes the
        list pessimistic (and hides a fix worth knowing about), while a corpus command
        that stopped crediting is the regression the whole module exists to catch. This
        is the invariant; the credit rate below is only a coarse trend on top of it.
        """
        commands = [c for c, _ in _CREDITING_CORPUS] + [c for c, _ in _KNOWN_UNCREDITED]
        measured_blind = {c for c in commands if not _credits_semantics(c)}
        self.assertEqual(measured_blind, {c for c, _ in _KNOWN_UNCREDITED})


#: A file every probe run starts with pending, so a whole-project validator has
#: something to clear and therefore something to show for itself.
_SENTINEL_PENDING = "pending.py"

#: A command that classifies cleanly and credits nothing — the control run.
_NOOP_COMMAND = "true"


def _files_named(command: str) -> tuple[str, ...]:
    """Paths *command* names, used to seed the dirty set for a credit probe.

    A validation credit is only ever given to an already-dirty file, so a fixed seed
    cannot serve every command: probing ``gcc src/mesh.c`` against a set holding only
    ``solver.py`` reports "credits nothing" for a pipeline that works fine. Seeding
    from the command's own operands removes that artifact.
    """
    segments = classify_bash_command(command)
    if not segments:
        return ()
    return tuple(
        os.path.normpath(op)
        for seg in segments for op in seg.operands
        if os.path.splitext(op)[1]
    )


def _semantic_view(ec: dict) -> tuple:
    """The content-bearing part of a context — what the command actually taught.

    ``action_op_count`` is excluded on purpose. Every EXEC segment bumps it, so
    ``make test`` and ``./solver --check`` would score as credited while having taught
    nothing about *what* they read, wrote or established — which is the whole question.
    Counting it inflated the measured rate from 82% to a flattering 90%.
    """
    return (
        frozenset(ec["read_files"]), bool(ec["searched"]), frozenset(ec["inspected_dirs"]),
        frozenset(ec["dirty_written_files"]), frozenset(ec["validated_files"]),
        frozenset(ec["runs"]),
        frozenset(ec["tests_run"]), tuple(ec.get("env_mutations") or ()),
        ec.get("last_edit_success_path") or "",
    )


def _credits_semantics(command: str) -> bool:
    """Did *command* teach the blackboard anything with content?

    The coarse question the per-command assertions refine. Measured by running the
    pipeline and diffing against a control run of a no-op command under the *same*
    seeding — not by counting list entries, which would report a healthy rate for a
    pipeline that had stopped working entirely.
    """
    seed = (_SENTINEL_PENDING, *_files_named(command))
    return _semantic_view(_run(command, dirty=seed)) != _semantic_view(
        _run(_NOOP_COMMAND, dirty=seed)
    )


class CreditRateTests(unittest.TestCase):
    """The number that makes a whole unhandled command family visible."""

    # Measured over the union of both lists — the realistic shell surface. The floor
    # sits close enough to the current rate that two new blind spots trip it, so
    # growing the blind surface is a decision someone takes rather than a drift. A
    # blind spot that starts crediting raises the rate and is caught by
    # BlindSurfaceTests instead, which is why that direction needs no ceiling here.
    MIN_CREDIT_RATE = 0.78

    def test_credit_rate_above_floor(self) -> None:
        commands = [c for c, _ in _CREDITING_CORPUS] + [c for c, _ in _KNOWN_UNCREDITED]
        credited = [c for c in commands if _credits_semantics(c)]
        rate = len(credited) / len(commands)
        uncredited = sorted(set(commands) - set(credited))
        self.assertGreaterEqual(
            rate, self.MIN_CREDIT_RATE,
            f"bash credit rate fell to {rate:.0%} ({len(credited)}/{len(commands)}). "
            f"Crediting nothing: {uncredited}. Either a command family stopped being "
            "credited, or the blind surface grew — both need a decision, not a lowered floor.",
        )


if __name__ == "__main__":
    unittest.main()
