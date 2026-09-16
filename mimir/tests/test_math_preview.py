"""Typeset calculations: the servers' ``latex`` field and the client's relay of it.

The math servers add a display-math body to their success payload; the client
detects it by shape and puts it on the ``tool_result`` event. No tool names here
either: the extractor only looks at the payload.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "servers", "_shared"))

from latex_display import number_latex, numeric_latex  # noqa: E402

from mimir.client.tool_execution.math_preview import (  # noqa: E402
    _MAX_LATEX_CHARS,
    extract_math_preview,
)


class NumericLatexTests(unittest.TestCase):
    def test_division_is_a_fraction(self) -> None:
        self.assertEqual(
            numeric_latex("(3 + 4) * 2 / sqrt(49)", 2.0),
            r"\frac{\left(3 + 4\right) \cdot 2}{\sqrt{49}} = 2",
        )

    def test_calculation_is_shown_as_asked(self) -> None:
        # SymPy would have evaluated the floor while parsing.
        self.assertEqual(numeric_latex("floor(3.7)", 3.0),
                         r"\left\lfloor 3.7 \right\rfloor = 3")

    def test_power_and_unary_minus(self) -> None:
        self.assertEqual(numeric_latex("-2**2", -4.0), "-{2}^{2} = -4")
        self.assertEqual(numeric_latex("(-2)**2", 4.0), r"{\left(-2\right)}^{2} = 4")
        self.assertEqual(numeric_latex("3 * -2", -6.0), r"3 \cdot \left(-2\right) = -6")

    def test_subtraction_keeps_its_grouping(self) -> None:
        self.assertEqual(numeric_latex("10 - (3 - 1)", 8.0), r"10 - \left(3 - 1\right) = 8")

    def test_rounded_result_is_approximate(self) -> None:
        self.assertEqual(numeric_latex("pi * 5 ** 2", 78.53981633974483),
                         r"\pi \cdot {5}^{2} \approx 78.5398163397")

    def test_number_formats(self) -> None:
        self.assertEqual(number_latex(1024.0), "1024")
        self.assertEqual(number_latex(3e20), r"3 \times 10^{20}")
        self.assertEqual(number_latex(float("inf")), r"\infty")

    def test_unparseable_expression_yields_nothing(self) -> None:
        self.assertIsNone(numeric_latex("2 +", 0.0))


class ExtractMathPreviewTests(unittest.TestCase):
    def test_latex_field_is_relayed(self) -> None:
        result = json.dumps({"status": "ok", "result": 4, "latex": "2 + 2 = 4"})
        self.assertEqual(extract_math_preview(result), {"latex": "2 + 2 = 4"})

    def test_payload_without_latex_has_no_preview(self) -> None:
        self.assertIsNone(extract_math_preview(json.dumps({"status": "ok", "result": 4})))
        self.assertIsNone(extract_math_preview("plain text"))

    def test_error_payload_has_no_preview(self) -> None:
        result = json.dumps({"status": "error", "error": "x", "latex": "1 = 1"})
        self.assertIsNone(extract_math_preview(result))

    def test_oversized_formula_is_dropped(self) -> None:
        result = json.dumps({"status": "ok", "latex": "x" * (_MAX_LATEX_CHARS + 1)})
        self.assertIsNone(extract_math_preview(result))


if __name__ == "__main__":
    unittest.main()
