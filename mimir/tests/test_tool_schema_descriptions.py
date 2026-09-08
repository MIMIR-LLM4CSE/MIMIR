"""Parameter descriptions are lifted out of the docstring and into the schema.

A constraint a tool states only in prose does not get followed. On one recorded run the
model dropped `report_verdict`'s required `verdict` twice, called `proxy_get` with an
`op` outside the six its docstring lists and with another the same docstring says needs
`name`, and hit `bash_run`'s 30s default four times against a docstring telling it to
raise the value — 9 wasted turns in 107 calls. The prose reached the model each time:
the whole docstring, `Args:` block included, is the tool's `description`. What gets
obeyed is the schema, and every parameter there was a bare `{"type": "string"}`.

So the text is not rewritten, it is moved to where it is read. These tests pin the
parser that moves it, including the shapes these docstrings are actually written in and
the malformed ones it has to survive — it runs at server registration, and a bad
docstring must never stop a tool being registered.
"""
import unittest

from mimir.client.integration.server_manager import (
    _args_block_descriptions,
    _schema_with_arg_descriptions,
)


class _Tool:
    def __init__(self, description, inputSchema):
        self.description = description
        self.inputSchema = inputSchema


def _schema(*names):
    return {"type": "object",
            "properties": {n: {"type": "string"} for n in names}}


class ArgsBlockParsingTests(unittest.TestCase):
    def test_a_plain_block(self) -> None:
        got = _args_block_descriptions(
            "Summary.\n\nArgs:\n    path: The file to read.\n    line: 1-based.\n"
        )
        self.assertEqual(got, {"path": "The file to read.", "line": "1-based."})

    def test_a_type_annotation_is_ignored(self) -> None:
        got = _args_block_descriptions("Args:\n    limit (int): How many.\n")
        self.assertEqual(got, {"limit": "How many."})

    def test_a_continuation_line_is_joined(self) -> None:
        got = _args_block_descriptions(
            "Args:\n    op: The operation,\n        which selects the rest.\n"
        )
        self.assertEqual(got, {"op": "The operation, which selects the rest."})

    def test_grouped_names_share_one_description(self) -> None:
        # How these docstrings are actually written: `run_a, run_b: ...`.
        got = _args_block_descriptions("Args:\n    run_a, run_b: Run IDs to diff.\n")
        self.assertEqual(got, {"run_a": "Run IDs to diff.", "run_b": "Run IDs to diff."})

    def test_entries_after_a_grouped_one_still_parse(self) -> None:
        # The regression this parser had: a grouped entry ended the block, silently
        # dropping every parameter documented after it.
        got = _args_block_descriptions(
            "Args:\n    a, b: Two things.\n    tail: Lines from the end.\n"
        )
        self.assertEqual(got.get("tail"), "Lines from the end.")

    def test_the_block_ends_at_the_next_section(self) -> None:
        got = _args_block_descriptions(
            "Args:\n    path: The file.\n\nReturns:\n    A dict with everything.\n"
        )
        self.assertEqual(got, {"path": "The file."})

    def test_no_args_block_yields_nothing(self) -> None:
        self.assertEqual(_args_block_descriptions("Just a summary line."), {})
        self.assertEqual(_args_block_descriptions(""), {})

    def test_an_entry_with_no_text_is_dropped(self) -> None:
        self.assertEqual(_args_block_descriptions("Args:\n    path:\n"), {})


class SchemaEnrichmentTests(unittest.TestCase):
    def test_missing_descriptions_are_filled(self) -> None:
        tool = _Tool("Args:\n    path: The file to read.\n", _schema("path"))
        out = _schema_with_arg_descriptions(tool)
        self.assertEqual(out["properties"]["path"]["description"], "The file to read.")

    def test_a_hand_written_description_always_wins(self) -> None:
        # A Field(description=...) on the server is a deliberate act; this is the
        # fallback for the parameters nobody got to.
        schema = _schema("verdict")
        schema["properties"]["verdict"]["description"] = "written by hand"
        tool = _Tool("Args:\n    verdict: from the docstring.\n", schema)
        out = _schema_with_arg_descriptions(tool)
        self.assertEqual(out["properties"]["verdict"]["description"], "written by hand")

    def test_the_servers_schema_is_never_mutated(self) -> None:
        # infer_tool_caps reads tool.inputSchema too, and must keep seeing exactly what
        # the server declared.
        schema = _schema("path")
        tool = _Tool("Args:\n    path: The file.\n", schema)
        _schema_with_arg_descriptions(tool)
        self.assertNotIn("description", schema["properties"]["path"])

    def test_a_name_the_schema_does_not_have_is_ignored(self) -> None:
        tool = _Tool("Args:\n    ghost: Not a parameter.\n", _schema("path"))
        out = _schema_with_arg_descriptions(tool)
        self.assertEqual(list(out["properties"]), ["path"])

    def test_a_tool_with_no_args_block_is_returned_unchanged(self) -> None:
        tool = _Tool("Just a summary.", _schema("path"))
        self.assertEqual(_schema_with_arg_descriptions(tool), _schema("path"))

    def test_a_malformed_schema_does_not_raise(self) -> None:
        # Registration must survive anything a server hands it.
        self.assertIsInstance(_schema_with_arg_descriptions(_Tool("Args:\n  a: b\n", None)), dict)
        self.assertIsInstance(
            _schema_with_arg_descriptions(_Tool(None, {"properties": "not a dict"})), dict
        )
        self.assertIsInstance(_schema_with_arg_descriptions(_Tool(None, {})), dict)


if __name__ == "__main__":
    unittest.main()
