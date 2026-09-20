"""The prompt is measured the way the provider measures it.

Regression cover for the accounting gap behind session ``b2fd38f1``: the trim and
compaction triggers sized history with ``content`` alone, while the request the
backend serialized carried ``tool_calls[].function.arguments`` too. On that run the
budgeteer saw ~44k tokens where the prompt held ~80k — of which 36k were tool-call
arguments, 22k of those ``write_file`` bodies — so neither trigger ever fired and
the force-fit backstop, which *did* count the arguments, had no way to reduce them.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mimir.client.event_sink import event_sink
from mimir.client.prompt.system_prompt import build_tool_catalog_for_planning
from mimir.client.query_engine import agent_loop
from mimir.client.query_engine import history as history_module
from mimir.client.query_engine.history import (
    _digest_call_args,
    _water_fill,
    bound_step_results,
    _digest_failed_call_args,
    _force_fit_to_window,
    _maybe_compact_intra_query,
    _message_tokens,
    WIRE_LIST_TOKENS_PER_MESSAGE,
    reconcile_tool_pairs,
)


def _call(call_id: str, name: str = "write_file", **args) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": dict(args)}}


def _assistant(*calls: dict, content=None) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def _ok(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps({"status": "ok", "operation": "updated"})}


def _err(call_id: str, error: str = "Write policy blocked tool 'write_file'.") -> dict:
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps({"status": "error", "error": error})}


def _content_only(messages: list[dict]) -> int:
    """The old, blind measure: content and nothing else."""
    return sum(len(m.get("content") or "") for m in messages)


class MessageTokensTests(unittest.TestCase):
    def test_counts_content_and_tool_call_arguments(self) -> None:
        m = _assistant(_call("c1", path="a.py", content="x" * 100), content="prose")
        self.assertGreater(_message_tokens(m, len), 100)

    def test_tolerates_content_none(self) -> None:
        """An assistant turn that only calls tools has ``content`` present but None.

        Both former call sites passed that None straight to the token counter.
        """
        m = _assistant(_call("c1", path="a.py"))
        self.assertIsNone(m["content"])
        self.assertGreater(_message_tokens(m, len), 0)

    def test_empty_content_still_costs_its_envelope(self) -> None:
        """A message with nothing to say is still a message on the wire.

        It used to score 0, which was the same optimism one layer down: the
        provider is sent ``{"role": "user", ...}`` whether or not the content is
        empty, and a budget that scores it free is a budget that will be short by
        one envelope per message. Small per message, and there are hundreds.
        """
        empty = _message_tokens({"role": "user", "content": ""}, len)
        self.assertGreater(empty, 0)
        # Bounded, though: an envelope is an envelope, not a message's worth.
        self.assertLess(empty, 32)

    def test_nothing_at_all_costs_nothing(self) -> None:
        self.assertEqual(_message_tokens({}, len), 0)


class WireUnitsTests(unittest.TestCase):
    """The budget's units are the provider's units.

    Regression cover for the crash that ended session ``7d322a3b``: the backstop
    announced ``truncated ~441,752 tokens of older content to fit the model's
    window`` — the branch it only takes when ``_force_fit_to_window`` returned
    True — and the very next call was refused with *Prompt (272,386 tokens)
    exceeds the model's context window (262,144 tokens)*. Both sides used the
    same tokenizer. They were counting different strings: the budget measured
    content and tool-call arguments as bare text, the provider measured
    ``json.dumps(messages)``. No tokenizer closes a difference in what is
    measured, so the fit was optimistic by construction — by ~18% on ordinary
    tool traffic and ~30% on the escape-dense pages a web fetch returns, which is
    what two of them had just put in the history.
    """

    @staticmethod
    def _escape_dense_history(n: int = 60) -> list[dict]:
        """The shape that broke it: quotes, newlines and JSON inside JSON."""
        fetched = json.dumps({
            "status": "ok",
            "body": "\n".join(
                f'<p class="mw-body">a "quoted" span \\( x_{i} \\) and a path /a/b_{i}.py</p>'
                for i in range(30)
            ),
        }, indent=2)
        messages = [{"role": "system", "content": "sys " * 200}]
        for i in range(n):
            messages.append(_assistant(_call(f"c{i}", url=f"https://x/{i}")))
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": fetched})
        return messages

    def test_budget_is_never_under_what_the_provider_will_count(self) -> None:
        messages = self._escape_dense_history()
        budget = (sum(_message_tokens(m, len) for m in messages)
                  + len(messages) * WIRE_LIST_TOKENS_PER_MESSAGE)
        # What VLLMBackend.chat measures before it refuses.
        on_the_wire = len(json.dumps(messages))
        self.assertGreaterEqual(
            budget, on_the_wire,
            "the budget under-counts what the provider will be sent — "
            "the shape that let a fit be announced on a prompt that was refused",
        )

    def test_the_old_text_shaped_measure_would_have_under_counted(self) -> None:
        """The premise, so the test above cannot pass by measuring nothing."""
        messages = self._escape_dense_history()
        text_shaped = sum(
            len("\n".join(
                [str(m.get("content") or "")]
                + [json.dumps((tc.get("function") or {}).get("arguments", ""))
                   for tc in (m.get("tool_calls") or [])]
            ))
            for m in messages
        )
        self.assertLess(text_shaped, len(json.dumps(messages)))

    def test_a_fitted_history_fits_on_the_wire(self) -> None:
        """End to end: what the backstop calls a fit, the provider can accept."""
        messages = self._escape_dense_history()
        window = len(json.dumps(messages)) // 2  # force the backstop to engage
        target = window - len(messages) * WIRE_LIST_TOKENS_PER_MESSAGE
        fitted = _force_fit_to_window(messages, target, len)
        self.assertTrue(fitted)
        self.assertLessEqual(len(json.dumps(messages)), window)


class CompactionTriggerTests(unittest.TestCase):
    """The regression from ``b2fd38f1``, reduced to a unit."""

    @staticmethod
    def _history() -> list[dict]:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        for i in range(4):
            messages.append(_assistant(_call(f"c{i}", path=f"f{i}.py", content="x" * 400)))
            messages.append(_ok(f"c{i}"))
        return messages

    def test_weight_in_arguments_is_invisible_to_the_content_only_measure(self) -> None:
        messages = self._history()
        budget = 1_000
        # The premise: under the old metric this history reads as comfortably small…
        self.assertLess(_content_only(messages), budget)
        # …while what actually goes on the wire is well over the budget.
        self.assertGreater(sum(_message_tokens(m, len) for m in messages), budget)

    def test_compaction_fires_on_a_history_whose_weight_is_in_arguments(self) -> None:
        messages = self._history()
        original = len(messages)

        def compact_fn(middle: list[dict]) -> list[dict]:
            return [{"role": "assistant", "content": "[summary]"}]

        with patch.object(history_module, "emit"):
            _maybe_compact_intra_query(
                messages, "sys", {}, compact_fn, token_counter=len, token_budget=1_000,
            )
        self.assertLess(len(messages), original)
        self.assertEqual(messages[2]["content"], "[summary]")

    def test_a_genuinely_small_history_is_still_left_alone(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        for i in range(4):
            messages.append(_assistant(_call(f"c{i}", path=f"f{i}.py")))
            messages.append(_ok(f"c{i}"))
        before = list(messages)
        _maybe_compact_intra_query(
            messages, "sys", {}, lambda middle: [{"role": "assistant", "content": "x"}],
            token_counter=len, token_budget=100_000,
        )
        self.assertEqual(messages, before)


class ForceFitArgumentsTests(unittest.TestCase):
    """The backstop must be able to reduce what it counts."""

    # Above the irreducible floor of the fixture below (~603 under the wire
    # measure: two protected messages plus every message's envelope). Was 600,
    # which sat just under that floor once the measure stopped ignoring the
    # envelope — the reduction under test still happens either way, so the
    # number is pressure, not the subject.
    TARGET = 700

    @staticmethod
    def _history() -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            _assistant(_call("old", path="big.py", content="x" * 4_000)),
            _ok("old"),
            _assistant(_call("new", path="small.py", content="y" * 10)),
            _ok("new"),
            {"role": "user", "content": "question"},
        ]

    def test_a_message_with_no_content_but_huge_arguments_is_reducible(self) -> None:
        messages = self._history()
        self.assertIsNone(messages[1]["content"])  # nothing for the old pass to shrink
        fitted = _force_fit_to_window(messages, self.TARGET, len)
        self.assertTrue(fitted)
        self.assertLessEqual(
            sum(_message_tokens(m, len) for m in messages), self.TARGET)
        self.assertIn("elided", messages[1]["tool_calls"][0]["function"]["arguments"]["content"])

    def test_the_newest_call_turn_keeps_its_arguments(self) -> None:
        """The turn whose results are being answered must stay legible to the model."""
        messages = self._history()
        _force_fit_to_window(messages, self.TARGET, len)
        self.assertEqual(messages[3]["tool_calls"][0]["function"]["arguments"]["content"],
                         "y" * 10)

    def test_still_reports_failure_when_the_core_alone_is_too_big(self) -> None:
        messages = [
            {"role": "system", "content": "s" * 5_000},
            {"role": "user", "content": "u" * 5_000},
        ]
        self.assertFalse(_force_fit_to_window(messages, 100, len))


class FailedCallDigestTests(unittest.TestCase):
    """Arguments of a call that never took effect are dead weight, not context."""

    @staticmethod
    def _history() -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            _assistant(_call("rejected", path="snap.py", content="x" * 2_000)),
            _err("rejected"),
            _assistant(_call("retry", path="snap.py", content="y" * 2_000)),
            _err("retry", "Syntax error in resulting Python content — file NOT written."),
            {"role": "user", "content": "q"},
        ]

    def test_an_older_failed_call_is_digested(self) -> None:
        messages = self._history()
        self.assertEqual(_digest_failed_call_args(messages), 1)
        args = messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIn("elided", args["content"])
        self.assertIn("2000 chars", args["content"])

    def test_the_newest_failure_keeps_its_arguments(self) -> None:
        """The model repairing a syntax error needs to see what it just wrote."""
        messages = self._history()
        _digest_failed_call_args(messages)
        self.assertEqual(messages[3]["tool_calls"][0]["function"]["arguments"]["content"],
                         "y" * 2_000)

    def test_a_successful_call_is_never_digested(self) -> None:
        messages = self._history()
        messages[2] = _ok("rejected")
        self.assertEqual(_digest_failed_call_args(messages), 0)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"]["content"],
                         "x" * 2_000)

    def test_digesting_twice_is_stable(self) -> None:
        messages = self._history()
        self.assertEqual(_digest_failed_call_args(messages), 1)
        self.assertEqual(_digest_failed_call_args(messages), 0)

    def test_scalar_arguments_survive_so_the_call_stays_readable(self) -> None:
        messages = self._history()
        messages[1]["tool_calls"][0]["function"]["arguments"]["overwrite"] = True
        _digest_failed_call_args(messages)
        args = messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args["path"], "snap.py")
        self.assertIs(args["overwrite"], True)


class DigestShapeTests(unittest.TestCase):
    def test_arguments_stay_a_dict(self) -> None:
        """A JSON string would be replaced by {} on the Anthropic path."""
        out = _digest_call_args({"path": "a.py", "content": "x" * 900})
        self.assertIsInstance(out, dict)

    def test_nothing_to_elide_returns_the_same_object(self) -> None:
        args = {"path": "a.py", "overwrite": True}
        self.assertIs(_digest_call_args(args), args)

    def test_a_json_string_payload_is_parsed_not_mangled(self) -> None:
        out = _digest_call_args(json.dumps({"path": "a.py", "content": "x" * 900}))
        self.assertIsInstance(out, dict)
        self.assertEqual(out["path"], "a.py")

    def test_call_ids_survive_so_pairing_still_reconciles(self) -> None:
        messages = FailedCallDigestTests._history()
        _digest_failed_call_args(messages)
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "rejected")
        self.assertEqual(reconcile_tool_pairs(list(messages)), messages)

    def test_the_source_arguments_are_never_mutated_in_place(self) -> None:
        """The archive shares these nested dicts; a digest must not reach into them."""
        messages = FailedCallDigestTests._history()
        original_call = messages[1]["tool_calls"][0]
        original_args = original_call["function"]["arguments"]
        _digest_failed_call_args(messages)
        self.assertEqual(original_args["content"], "x" * 2_000)
        self.assertIsNot(messages[1]["tool_calls"][0], original_call)


class StepResultBudgetTests(unittest.TestCase):
    """What one step may add to the context, bounded before it is appended.

    Cover for the session that went from 200k to 215k in a single step: four fetches
    landed together, one of them a 131k-token HTTP 403 body. Nothing looked at the
    sum. The trim pass exempts the current step from eviction, so the only thing that
    could have reduced them was the force-fit backstop — and only once the window was
    already over.
    """

    # One token per character keeps the arithmetic in these tests readable.
    _TOK = staticmethod(len)

    def _bound(self, results, mode="full"):
        return bound_step_results(
            results, model="m", context_mode=mode, token_counter=self._TOK)

    def _budget(self, mode="full"):
        from mimir.client.config.constants import context_budget_for
        total, reserved, _t, _c = context_budget_for("m", mode)
        return total - reserved

    def test_a_result_within_budget_is_untouched(self):
        out, cut = self._bound(["small", "also small"])
        self.assertEqual(out, ["small", "also small"])
        self.assertEqual(cut, [])

    def test_one_oversized_result_is_cut_head_and_tail_and_says_so(self):
        usable = self._budget()
        huge = "A" * (usable // 2) + "OMEGA"
        out, cut = self._bound([huge])
        self.assertLess(len(out[0]), len(huge))
        self.assertIn("TOOL_RESULT_TRUNCATED", out[0])
        self.assertIn(str(len(huge)), out[0])       # names the real size
        self.assertIn("[truncated]", out[0])        # head+tail marker, not a head cut
        self.assertTrue(out[0].startswith("AAA"))   # head kept
        self.assertEqual(cut[0][0], len(huge))
        self.assertLess(cut[0][1], cut[0][0])

    def test_four_parallel_results_are_bounded_by_their_sum(self):
        """The 215k shape: each call reasonable on its own, the step is not."""
        usable = self._budget()
        results = ["X" * (usable // 3) for _ in range(4)]
        out, cut = self._bound(results)
        self.assertLessEqual(sum(len(r) for r in out), usable)
        self.assertEqual(len(cut), 4)

    def test_a_small_result_is_not_cut_to_pay_for_a_large_one(self):
        usable = self._budget()
        out, _cut = self._bound(["tiny", "Z" * usable, "also tiny"])
        self.assertEqual(out[0], "tiny")
        self.assertEqual(out[2], "also tiny")
        self.assertIn("TOOL_RESULT_TRUNCATED", out[1])

    def test_the_budget_follows_the_window_instead_of_being_a_fixed_number(self):
        """A ceiling that is generous on 200k is catastrophic on 32k."""
        full = self._budget("full")
        compact = self._budget("compact")
        self.assertGreater(full, compact)
        text = "Q" * (compact * 2)
        out_full, _ = self._bound([text], mode="full")
        out_compact, _ = self._bound([text], mode="compact")
        self.assertLess(len(out_compact[0]), len(out_full[0]))

    def test_the_note_fits_inside_the_allowance_it_explains(self):
        """Bounding the result must not itself be what breaks the bound."""
        usable = self._budget()
        out, _cut = self._bound(["Y" * (usable * 3)])
        self.assertLessEqual(len(out[0]), usable)

    def test_an_empty_step_is_not_a_special_case(self):
        self.assertEqual(self._bound([]), ([], []))


class BackstopIsWideEnoughTests(unittest.TestCase):
    """A backstop that fires in normal operation is not a backstop.

    The shares started at usable//4, which on a 160k usable window is 40k tokens —
    and `github_get_file` is allowed to return 256 KB, about 65k. A file fetch doing
    exactly what it is designed to do would have been cut, silently, by the pass
    meant to catch pathological results. These numbers are the ceilings MIMIR's own
    servers declare; if one of them rises past the share, the share moves, not the
    result.
    """

    # (tool, its own documented maximum, in bytes)
    _SERVER_CEILINGS = [
        ("bash_run", 64 * 1024),              # server_bash._MAX_OUTPUT
        ("slurm/hpc", 128 * 1024),            # server_hpc._MAX_OUTPUT
        ("http_get prose", 128 * 1024),       # server_web._MAX_TEXT_CHARS
        ("github_get_file", 256 * 1024),      # server_github._MAX_FILE_BYTES
    ]
    _CHARS_PER_TOKEN = 4                      # constants.CHARS_PER_TOKEN

    def test_no_shipped_tool_is_cut_at_its_own_maximum(self):
        from mimir.client.config.constants import context_budget_for
        from mimir.client.query_engine import history as H

        total, reserved, _t, _c = context_budget_for("m", "full")
        usable = total - reserved
        per_result = usable // H._RESULT_WINDOW_SHARE
        for name, ceiling_bytes in self._SERVER_CEILINGS:
            with self.subTest(tool=name):
                self.assertLessEqual(
                    ceiling_bytes // self._CHARS_PER_TOKEN, per_result,
                    f"{name} at its own ceiling would be cut by the backstop")

    def test_a_step_may_still_not_take_the_whole_window(self):
        """Wide is not unbounded: the step share has to leave room for the history
        the results are being added to."""
        from mimir.client.query_engine import history as H
        self.assertLess(H._STEP_NUM / H._STEP_DEN, 1.0)

    def test_the_original_incident_is_still_caught(self):
        """The four fetches that took a 200k window to 215k: a 131k-token error body
        beside three ordinary ones."""
        from mimir.client.config.constants import context_budget_for

        total, reserved, _t, _c = context_budget_for("m", "full")
        usable = total - reserved
        results = ["A" * (131_164 * 4), "B" * (34_479 * 4),
                   "C" * (3_366 * 4), "D" * (3_142 * 4)]
        out, cut = bound_step_results(
            results, model="m", context_mode="full", token_counter=len)
        self.assertTrue(cut)
        self.assertLessEqual(sum(len(r) for r in out), usable * 4)


class WaterFillTests(unittest.TestCase):
    def test_everything_fits_when_the_budget_allows(self):
        self.assertEqual(_water_fill([10, 20, 30], 100), [10, 20, 30])

    def test_the_budget_released_by_small_items_raises_the_rest(self):
        # A flat share would be 10 each; the two small ones release 17 for the big one.
        allow = _water_fill([1, 2, 100], 30)
        self.assertEqual(allow[0], 1)
        self.assertEqual(allow[1], 2)
        self.assertEqual(allow[2], 27)
        self.assertLessEqual(sum(allow), 30)

    def test_equal_items_split_the_budget_evenly(self):
        self.assertEqual(_water_fill([50, 50], 30), [15, 15])

    def test_it_never_exceeds_the_budget(self):
        for budget in (0, 1, 7, 999):
            self.assertLessEqual(sum(_water_fill([3, 30, 300], budget)), budget)


class EnforceBudgetIntegrationTests(unittest.TestCase):
    def test_failed_arguments_are_dropped_regardless_of_the_budget(self) -> None:
        messages = FailedCallDigestTests._history()
        emitted: list[dict] = []
        with patch.object(history_module, "emit", emitted.append), \
             patch.object(history_module, "context_budget_for",
                          lambda model, mode: (1_000_000, 200_000, 500_000, 400_000)):
            history_module._enforce_context_budget(
                messages, "sys", None, {}, "m", "full", None, lambda t: len(t) // 4,
            )
        self.assertIn("elided", messages[1]["tool_calls"][0]["function"]["arguments"]["content"])
        self.assertTrue(any("failed tool call" in e.get("text", "") for e in emitted))


if __name__ == "__main__":
    unittest.main()


class InterruptedQueryStateTests(unittest.TestCase):
    """A query that dies still says what it left behind.

    Session ``7d322a3b`` ended on a provider error rendered as the entire answer.
    A source file edited ninety seconds earlier had never been measured and an
    optimisation session sat open on a baseline nobody would remember — none of
    which appeared anywhere, because the ledger that reports exactly this is
    built on the normal exit path and that path was never reached.
    """

    @staticmethod
    def _context_with_unfinished_work() -> dict:
        return {
            "dirty_written_files": {"/w/wave2d/abc.py"},
            "validated_files": set(),
            "runs": {},
        }

    def test_state_is_emitted_when_a_query_dies(self) -> None:
        events: list[dict] = []
        with event_sink(events.append):
            agent_loop._emit_interrupted_state(self._context_with_unfinished_work())
        text = " ".join(e.get("text", "") for e in events)
        self.assertIn("ended early", text)
        self.assertIn("abc.py", text)

    def test_a_clean_context_says_nothing(self) -> None:
        events: list[dict] = []
        with event_sink(events.append):
            agent_loop._emit_interrupted_state({})
        self.assertEqual(events, [])

    def test_reporting_never_masks_the_failure_it_reports(self) -> None:
        """It runs with an exception in flight; it may not raise a second one."""
        events: list[dict] = []
        with event_sink(events.append):
            agent_loop._emit_interrupted_state({"dirty_written_files": object()})  # type: ignore[dict-item]


class PlanModeToolCatalogTests(unittest.TestCase):
    """Plan mode hides the tools; it must not hide what they do.

    In session ``7d322a3b`` the plan was built entirely around ``proxy_eval`` /
    ``proxy_manage`` / ``proxy_exec``, all of which are PLAN_BLOCKED and therefore
    stripped from the tool list — schemas included. A third of the planning phase
    went on recovering their contract: two filesystem-wide ``find`` calls that timed
    out at 30 s and 60 s, a grep through the session's own transcripts, and finally
    ``sed -n '204,300p'`` over mimir's own server source. The docstrings enumerate
    every op; they were simply not in the room.
    """

    _DESCRIPTIONS = {
        "proxy_eval": ("Drive an iterative proxy-optimization session as a monotone "
                       "ratchet (sensitive).\n\n    Operations (set op): init, run"),
        "bash_run": "Run a shell command in the workspace.",
    }

    def _catalog(self, **kw):
        return build_tool_catalog_for_planning(
            {"proxy_eval": "proxy", "bash_run": "bash"}, {"proxy_eval"}, **kw)

    def test_each_tool_carries_what_it_does(self) -> None:
        catalog = self._catalog(tool_descriptions=self._DESCRIPTIONS)
        self.assertIn("monotone ratchet", catalog)
        self.assertIn("shell command", catalog)

    def test_the_name_and_its_server_are_still_there(self) -> None:
        catalog = self._catalog(tool_descriptions=self._DESCRIPTIONS)
        self.assertIn("proxy_eval", catalog)
        self.assertIn("- proxy:", catalog)
        self.assertIn("[sensitive]", catalog)

    def test_one_line_per_tool_and_no_more(self) -> None:
        """A catalog long enough to be skipped is the same as no catalog."""
        catalog = self._catalog(tool_descriptions=self._DESCRIPTIONS)
        for line in catalog.splitlines():
            self.assertLessEqual(len(line), 140, line)
        # The op list belongs to the schema, not here.
        self.assertNotIn("Operations (set op)", catalog)

    def test_it_degrades_to_names_when_no_description_is_known(self) -> None:
        catalog = self._catalog()
        self.assertIn("proxy_eval", catalog)
        self.assertNotIn("—", catalog)
