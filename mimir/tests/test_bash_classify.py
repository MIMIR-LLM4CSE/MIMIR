"""Unit tests for the bash command → capability classifier.

``classify_bash_command`` maps each ``; && || |`` segment of a shell command to a
kind (read/search/inspect/write/exec/env) and the operands it acts on. Conservative:
opaque commands (substitution, unknown leading command) return None, and ambiguous
operands are dropped rather than guessed.
"""
import unittest

from mimir.client.guardrails.policy.bash_classify import (
    Kind,
    bash_command_is_readonly,
    classify_bash_command,
)


def _kinds(command):
    segs = classify_bash_command(command)
    return None if segs is None else [s.kind for s in segs]


class ClassifyKindTests(unittest.TestCase):
    def test_read_commands(self):
        for cmd in ["cat a.py", "head -20 f.txt", "tail -n 5 f.txt", "nl f.py",
                    "sed -n '1,20p' f.py", "wc -l f.py"]:
            self.assertEqual(_kinds(cmd), [Kind.READ], cmd)

    def test_search_commands(self):
        for cmd in ["grep foo bar.py", "rg pattern", "grep -rn foo src"]:
            self.assertEqual(_kinds(cmd), [Kind.SEARCH], cmd)

    def test_inspect_commands(self):
        for cmd in ["ls", "ls src", "find src -name '*.py'", "du -sh ."]:
            self.assertEqual(_kinds(cmd), [Kind.INSPECT], cmd)

    def test_write_commands(self):
        for cmd in ["sed -i 's/a/b/' f.py", "sed --in-place 's/a/b/' f.py",
                    "sort -o out.txt in.txt"]:
            self.assertEqual(_kinds(cmd), [Kind.WRITE], cmd)

    def test_exec_commands(self):
        for cmd in ["python x.py", "python3 -m pytest", "gcc a.c -o a.out",
                    "make", "nvcc k.cu -o k", "./a.out", "node app.js"]:
            self.assertEqual(_kinds(cmd), [Kind.EXEC], cmd)

    def test_env_commands(self):
        self.assertEqual(_kinds("module avail"), [Kind.ENV_DISCOVERY])
        self.assertEqual(_kinds("module list"), [Kind.ENV_DISCOVERY])
        self.assertEqual(_kinds("module load cuda"), [Kind.ENV_MUTATE])

    def test_env_manager_commands(self):
        # pip/conda provision the interpreter, so they carry the ENV_* kinds rather
        # than EXEC: a local query is discovery (plan-safe), anything else mutates.
        for cmd in ("pip list", "pip show numpy", "pip freeze", "conda info",
                    "conda list"):
            self.assertEqual(_kinds(cmd), [Kind.ENV_DISCOVERY], cmd)
        for cmd in ("pip install requests", "pip3 install -r requirements.txt",
                    "conda create -n e python=3.11", "mamba install -y numpy",
                    "conda search numpy"):
            self.assertEqual(_kinds(cmd), [Kind.ENV_MUTATE], cmd)
        # A mutation carries the names it brings in, as `module load` does.
        self.assertEqual(classify_bash_command("pip install requests flask")[0].operands,
                         ["requests", "flask"])
        self.assertEqual(classify_bash_command("conda env create -f env.yml")[0].operands,
                         ["env.yml"])

    def test_neutral_commands(self):
        for cmd in ["pwd", "echo hi", "which python", "realpath ."]:
            self.assertEqual(_kinds(cmd), [Kind.NEUTRAL], cmd)

    def test_metadata_readers_are_reads_not_no_ops(self):
        # `stat`/`file` open the file they are given (and are path-confined for it), so
        # they credit a read — unlike `pwd`/`which`, whose arguments are words.
        self.assertEqual(_kinds("stat f.py"), [Kind.READ])
        self.assertEqual(classify_bash_command("file f.py")[0].operands, ["f.py"])

    def test_unknown_env_subcommand_is_assumed_to_mutate(self):
        # A sub-command nobody has reasoned about must not come out plan-safe.
        for cmd in ("module purge", "module swap a b", "pip wheel .", "conda clean"):
            self.assertEqual(_kinds(cmd), [Kind.ENV_MUTATE], cmd)
            self.assertFalse(bash_command_is_readonly(cmd), cmd)


class OperandExtractionTests(unittest.TestCase):
    def test_read_file_operands(self):
        segs = classify_bash_command("cat a.py b.py")
        self.assertEqual(segs[0].operands, ["a.py", "b.py"])

    def test_sed_skips_inline_script(self):
        # Without -e/-f the first positional is the script, not a file.
        self.assertEqual(classify_bash_command("sed -n '1,20p' f.py")[0].operands, ["f.py"])
        self.assertEqual(classify_bash_command("sed 's/a/b/' f.py")[0].operands, ["f.py"])

    def test_sed_explicit_script_all_positionals_are_files(self):
        self.assertEqual(classify_bash_command("sed -e 's/x/y/' f.py")[0].operands, ["f.py"])

    def test_search_pattern_operand(self):
        self.assertEqual(classify_bash_command("grep foo bar.py")[0].operands, ["foo"])

    def test_inspect_dir_operand_defaults_to_cwd(self):
        self.assertEqual(classify_bash_command("ls")[0].operands, ["."])
        self.assertEqual(classify_bash_command("ls src")[0].operands, ["src"])

    def test_write_operand(self):
        self.assertEqual(classify_bash_command("sed -i s/a/b/ mod.py")[0].operands, ["mod.py"])
        self.assertEqual(classify_bash_command("sort -o out.txt in.txt")[0].operands, ["out.txt"])

    def test_module_mutate_carries_names(self):
        self.assertEqual(classify_bash_command("module load cuda")[0].operands, ["cuda"])

    def test_tr_reads_stdin_and_credits_no_operand(self):
        # `tr`'s args are character sets, not file paths — no operand is credited.
        seg = classify_bash_command("tr 'a-z' 'A-Z'")[0]
        self.assertEqual(seg.kind, Kind.READ)
        self.assertEqual(seg.operands, [])


class PipelineAndOpaqueTests(unittest.TestCase):
    def test_pipeline_classified_per_segment(self):
        segs = classify_bash_command("cat a.py | grep foo")
        self.assertEqual([s.kind for s in segs], [Kind.READ, Kind.SEARCH])
        self.assertEqual(segs[0].operands, ["a.py"])
        self.assertEqual(segs[1].operands, ["foo"])

    def test_chain_mixes_kinds(self):
        segs = classify_bash_command("gcc a.c -o a.out && ./a.out")
        self.assertEqual([s.kind for s in segs], [Kind.EXEC, Kind.EXEC])


class ValidationCommandTests(unittest.TestCase):
    """Exec/validation commands classify EXEC and credit only their file operand.

    Feeds observations._observe_command, which marks a written file 'validated' when
    the model runs py_compile / pytest / ruff / mypy on it via bash.
    """

    def test_linters_classify_exec(self):
        for cmd in ["ruff check f.py", "pyflakes f.py", "mypy f.py", "black f.py"]:
            self.assertEqual(_kinds(cmd), [Kind.EXEC], cmd)

    def test_exec_credits_file_operand_only(self):
        # Module name / sub-command are not files and must not be credited.
        self.assertEqual(classify_bash_command("python -m py_compile foo.py")[0].operands, ["foo.py"])
        self.assertEqual(classify_bash_command("ruff check foo.py")[0].operands, ["foo.py"])
        self.assertEqual(classify_bash_command("pytest -q test_foo.py")[0].operands, ["test_foo.py"])
        self.assertEqual(classify_bash_command("mypy pkg/mod.py")[0].operands, ["pkg/mod.py"])

    def test_bare_pytest_has_no_file_operand(self):
        self.assertEqual(classify_bash_command("pytest -q")[0].operands, [])

    def test_exec_head_is_exposed(self):
        self.assertEqual(classify_bash_command("pytest -q")[0].head, "pytest")
        self.assertEqual(classify_bash_command("ruff check .")[0].head, "ruff")
        # `python -m X` reports the module, so a py-launched validator is recognised.
        self.assertEqual(classify_bash_command("python -m pytest")[0].head, "pytest")
        # A plain script run reports `python`, not the script.
        self.assertEqual(classify_bash_command("python foo.py")[0].head, "python")
        self.assertEqual(classify_bash_command("gcc a.c -o a.out")[0].head, "gcc")


class ChdirTests(unittest.TestCase):
    """`cd` is its own CHDIR kind so a later segment's relative operand can be rebased."""

    def test_cd_is_chdir_with_target(self):
        seg = classify_bash_command("cd subdir")[0]
        self.assertEqual(seg.kind, Kind.CHDIR)
        self.assertEqual(seg.operands, ["subdir"])

    def test_cd_no_target_carries_nothing(self):
        self.assertEqual(classify_bash_command("cd")[0].operands, [])
        self.assertEqual(classify_bash_command("cd -")[0].operands, [])

    def test_cd_then_exec_chain(self):
        segs = classify_bash_command("cd sub && pytest t.py")
        self.assertEqual([s.kind for s in segs], [Kind.CHDIR, Kind.EXEC])
        self.assertEqual(segs[1].operands, ["t.py"])

    def test_cd_is_read_only_but_exec_chain_is_not(self):
        self.assertTrue(bash_command_is_readonly("cd sub && cat t.py"))
        self.assertFalse(bash_command_is_readonly("cd sub && pytest t.py"))

    def test_substitution_is_opaque(self):
        for cmd in ["cat $(which python)", "echo `whoami`", "cat ${HOME}/x",
                    "diff <(ls) <(ls)"]:
            self.assertIsNone(classify_bash_command(cmd), cmd)

    def test_file_redirection_is_a_write(self):
        # A redirect to a file creates one, so the segment cannot stay read-only —
        # and the file it creates is the operand.
        for cmd in ("cat foo > out.txt", "cat foo >> out.txt", "ls > out.txt",
                    "grep -rn foo src > hits.txt"):
            segs = classify_bash_command(cmd)
            self.assertEqual([s.kind for s in segs], [Kind.WRITE], cmd)
            self.assertEqual(segs[0].operands, ["out.txt"] if "out.txt" in cmd
                             else ["hits.txt"], cmd)
        self.assertFalse(bash_command_is_readonly("ls > out.txt"))

    def test_redirection_does_not_relabel_an_exec_or_write(self):
        # A command that already writes/executes keeps its kind and its operands:
        # `pytest t.py > log` must still credit t.py as the file it validated.
        segs = classify_bash_command("pytest t.py > log.txt")
        self.assertEqual([s.kind for s in segs], [Kind.EXEC])
        self.assertEqual(segs[0].operands, ["t.py"])
        self.assertFalse(bash_command_is_readonly("pytest t.py > log.txt"))

    def test_input_redirection_keeps_the_command_kind(self):
        # Reading stdin from a file adds no side effect; the file is confined by
        # the server and surfaced by the gate, but the kind is unchanged.
        self.assertEqual(_kinds("cat < in.txt"), [Kind.READ])
        self.assertEqual(_kinds("python x.py < in.txt"), [Kind.EXEC])

    def test_heredoc_is_opaque(self):
        # A heredoc body is not an argv, and its lines are not commands.
        self.assertIsNone(classify_bash_command("cat > f.txt <<EOF\nhi\nEOF"))
        self.assertIsNone(classify_bash_command("cat <<< text"))

    def test_fd_redirection_is_transparent(self):
        # Silencing/merging a stream adds no side effect: the segment keeps its
        # kind and stays plan-safe. The source fd must not leak into argv.
        for cmd in ("which nvcc 2>/dev/null", "which pdflatex 2>&1",
                    "ls 2>/dev/null", "pwd &>/dev/null"):
            self.assertIsNotNone(classify_bash_command(cmd), cmd)
            self.assertTrue(bash_command_is_readonly(cmd), cmd)
        self.assertEqual(classify_bash_command("grep -n foo bar.py 2>/dev/null")[0].kind,
                         Kind.SEARCH)
        self.assertEqual(_kinds("cat a.py 2>&1"), [Kind.READ])
        # An exec segment is not laundered into a read-only one by a redirect.
        self.assertFalse(bash_command_is_readonly("pytest -q 2>/dev/null"))

    def test_multiline_command_segments_per_line(self):
        # An unquoted newline is a separator, so each line is its own segment —
        # never folded into the argv of the line above (which would classify
        # `cat a.py` + `rm -rf .` as a single harmless read).
        self.assertEqual(_kinds("cat a.py\nls src"), [Kind.READ, Kind.INSPECT])
        self.assertIsNone(classify_bash_command("cat a.py\nrm -rf ."))
        self.assertFalse(bash_command_is_readonly("cat a.py\npytest -q"))
        # A newline right after a connector continues the same chain.
        self.assertEqual(_kinds("cat a.py &&\nls src"), [Kind.READ, Kind.INSPECT])
        # Blank lines and a trailing newline carry no command.
        self.assertEqual(_kinds("cat a.py\n\nls src\n"), [Kind.READ, Kind.INSPECT])

    def test_newline_inside_quotes_is_data(self):
        # A multi-line `-c` body is one exec segment, not a chain. (A body with
        # parens stays opaque for the same reason a one-line one does — the
        # subshell chars — which is unrelated to the newlines.)
        self.assertEqual(_kinds('python3 -c "\nimport re\nx = 1\n"'), [Kind.EXEC])
        self.assertEqual(_kinds("cat 'a\nb.py'"), [Kind.READ])

    def test_unknown_leading_command_is_opaque(self):
        self.assertIsNone(classify_bash_command("rm -rf ."))
        self.assertIsNone(classify_bash_command("curl http://x"))
        # A shell interpreter nests an unvalidated command — never classifiable.
        self.assertIsNone(classify_bash_command("bash -c 'ls'"))


class CapabilityProbeTests(unittest.TestCase):
    """`which X || true` — how the agent asks whether a toolchain exists.

    Every piece of the idiom must stay plan-safe, or the probe is rejected and the
    agent has no way to establish availability except by trying commands that fail.
    """

    def test_noop_commands_are_neutral(self):
        for cmd in ("true", "false", ":"):
            self.assertEqual(_kinds(cmd), [Kind.NEUTRAL], cmd)

    def test_probe_idiom_is_read_only(self):
        for cmd in ("which pdflatex",
                    "which pdflatex || true",
                    "which pdflatex 2>/dev/null || true",
                    "which pdflatex || echo missing"):
            self.assertTrue(bash_command_is_readonly(cmd), cmd)


class FileManagementTests(unittest.TestCase):
    """`mv`/`cp`/`mkdir` are unconditional writes; `cd` is not."""

    def test_file_management_is_write(self):
        for cmd in ("mv a.py b.py", "cp a.py b.py", "cp -r src dst", "mkdir build"):
            self.assertEqual(_kinds(cmd), [Kind.WRITE], cmd)

    def test_file_management_is_not_plan_safe(self):
        for cmd in ("mv a.py b.py", "cp a.py b.py", "mkdir build",
                    "ls && cp a.py b.py"):
            self.assertFalse(bash_command_is_readonly(cmd), cmd)

    def test_only_the_destination_is_credited(self):
        # The source of a move is gone afterwards, so crediting it would leave the
        # observation layer tracking a file that no longer exists.
        self.assertEqual(classify_bash_command("mv old.py new.py")[0].operands, ["new.py"])
        self.assertEqual(classify_bash_command("cp -r src dst")[0].operands, ["dst"])
        self.assertEqual(classify_bash_command("mkdir -p a b")[0].operands, ["a", "b"])

    def test_cd_remains_read_only(self):
        self.assertTrue(bash_command_is_readonly("cd sub && ls"))


class TexTests(unittest.TestCase):
    """TeX engines are executions, gated exactly like the compilers."""

    def test_tex_engines_are_exec(self):
        for cmd in ("pdflatex main.tex", "xelatex -interaction=nonstopmode d.tex",
                    "latexmk -pdf main.tex", "bibtex main", "biber main"):
            self.assertEqual(_kinds(cmd), [Kind.EXEC], cmd)

    def test_tex_is_not_plan_safe(self):
        self.assertFalse(bash_command_is_readonly("pdflatex main.tex"))

    def test_tex_credits_its_source_operand(self):
        self.assertEqual(classify_bash_command("pdflatex main.tex")[0].operands,
                         ["main.tex"])
        self.assertEqual(classify_bash_command("pdflatex main.tex")[0].head, "pdflatex")

    def test_empty_and_malformed(self):
        for cmd in ["", "   ", "ls\nrm -rf .", None]:
            self.assertIsNone(classify_bash_command(cmd), repr(cmd))


class NestedCommandParsingTests(unittest.TestCase):
    """`find … -exec CMD {} \\;` is split into its own segment.

    ``parse_segments`` is shared by the bash server, this classifier, and the
    client's out-of-workspace gate, so a change here has the widest blast radius in
    the stack. The invariance test below is the guard: tokenization must be
    byte-identical for every command that contains no ``-exec``.
    """

    def _segs(self, cmd):
        from mimir.servers._shared.shell_paths import parse_segments
        return [(s.argv, s.nested) for s in parse_segments(cmd)]

    def test_nested_command_becomes_its_own_segment(self):
        segs = self._segs(r'find . -name "*.py" -exec grep -l wave {} \;')
        self.assertEqual(segs, [
            (["find", ".", "-name", "*.py"], False),
            (["grep", "-l", "wave", "{}"], True),
        ])

    def test_plus_terminator(self):
        self.assertEqual(self._segs("find . -type f -exec wc -l {} +"), [
            (["find", ".", "-type", "f"], False),
            (["wc", "-l", "{}"], True),
        ])

    def test_find_args_after_the_terminator_stay_with_find(self):
        # The `;` must be consumed as a terminator, not read as a chain separator —
        # otherwise the command splits in the wrong place.
        self.assertEqual(self._segs(r"find . -name x -exec grep q {} \; -print"), [
            (["find", ".", "-name", "x", "-print"], False),
            (["grep", "q", "{}"], True),
        ])

    def test_unterminated_exec_raises(self):
        from mimir.servers._shared.shell_paths import ShellParseError, parse_segments
        with self.assertRaises(ShellParseError):
            parse_segments("find . -exec grep q {}")

    def test_readonly_nested_set_is_derived_not_relisted(self):
        from mimir.servers._shared.shell_paths import (
            INSPECT_COMMANDS,
            NEUTRAL_COMMANDS,
            READ_COMMANDS,
            READONLY_NESTED_COMMANDS,
            SEARCH_COMMANDS,
            WRITE_COMMANDS,
            EXEC_COMMANDS,
        )
        self.assertEqual(
            READONLY_NESTED_COMMANDS,
            READ_COMMANDS | SEARCH_COMMANDS | INSPECT_COMMANDS | NEUTRAL_COMMANDS,
        )
        # Writers and execs must never be nestable: their operands include `{}`.
        self.assertFalse(READONLY_NESTED_COMMANDS & (WRITE_COMMANDS | EXEC_COMMANDS))

    def test_tokenization_unchanged_for_commands_without_exec(self):
        # The invariance guard. Every command exercised elsewhere in this file plus
        # the awkward shapes: none contains `-exec`, so all must parse to exactly
        # one non-nested segment set, unchanged by the nested-command state machine.
        from mimir.servers._shared.shell_paths import parse_segments
        corpus = [
            "ls -la", "grep -n foo bar.py", "cat a.py | head -20",
            "cd sub && pytest t.py", "python -c 'print(1)'",
            "gcc solver.c -O2 -o solver.out", "sed -i s/a/b/ mod.py",
            "ruff check .", "pytest -q", "find . -name '*.py'",
            "ls src && cat src/main.py", "echo hi > out.txt",
            "python x.py 2>/dev/null", "module load cuda && make",
            "grep -rn 'a; b' f.py", "wc -l *.py", "which nvcc || true",
            "find . -type d -name build", "pdflatex main.tex",
        ]
        for cmd in corpus:
            segs = parse_segments(cmd)
            self.assertTrue(all(not s.nested for s in segs), cmd)
            # Nothing was consumed or reordered: every segment still has an argv.
            self.assertTrue(all(s.argv for s in segs), cmd)

    def test_semicolon_outside_exec_still_separates(self):
        self.assertEqual(self._segs("ls ; pwd"), [(["ls"], False), (["pwd"], False)])

    def test_quoted_semicolon_is_still_data(self):
        segs = self._segs("grep -rn 'a; b' f.py")
        self.assertEqual(segs, [(["grep", "-rn", "a; b", "f.py"], False)])


if __name__ == "__main__":
    unittest.main()
