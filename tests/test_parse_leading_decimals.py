"""Regression tests for answers written without a zero before the decimal."""

import json
from pathlib import Path
import re
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from grader_utils.math_grader import grade_answer  # noqa: E402
from grader_utils.parse_utils import (  # noqa: E402
    _normalize,
    parse_answer_robust,
)


class LeadingDecimalParsingTest(unittest.TestCase):
    def test_math500_leading_decimal_answers_round_trip(self):
        dataset_path = REPOSITORY_ROOT / "data" / "MATH500.json"
        with dataset_path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)

        leading_decimals = {
            row["id"]: row["answer"]
            for row in dataset
            if re.fullmatch(r"\.\d+", row["answer"])
        }
        self.assertEqual(
            leading_decimals,
            {
                "test/number_theory/598.json": ".0000672",
                "test/number_theory/410.json": ".35625",
            },
        )

        for problem_id, ground_truth in leading_decimals.items():
            with self.subTest(problem_id=problem_id):
                parsed, source = parse_answer_robust(
                    rf"Therefore, \boxed{{{ground_truth}}}.",
                    return_source=True,
                )
                self.assertEqual(parsed, ground_truth)
                self.assertEqual(source, "boxed")
                self.assertTrue(grade_answer(parsed, ground_truth))

    def test_every_numeric_extraction_path_accepts_leading_decimals(self):
        cases = [
            (r"\boxed{.35625}", ".35625", "boxed"),
            ("Final Answer: .35625.", ".35625", "final_line"),
            ("```output\n.35625\n```", ".35625", "output_block"),
            ("print(value)\n.35625", ".35625", "print_output"),
            ("Thus, $ .35625 $.", ".35625", "inline_math"),
            ("The probability is .35625.", ".35625", "is_pattern"),
            (".35625", ".35625", "fallback_number"),
        ]

        for completion, expected, expected_source in cases:
            with self.subTest(source=expected_source):
                self.assertEqual(
                    parse_answer_robust(completion, return_source=True),
                    (expected, expected_source),
                )

    def test_normalization_removes_only_trailing_periods(self):
        self.assertEqual(_normalize(".0000672"), ".0000672")
        self.assertEqual(_normalize(".35625."), ".35625")
        self.assertEqual(_normalize("-.5."), "-.5")
        self.assertEqual(_normalize("42."), "42")
        self.assertEqual(_normalize("..."), "")

    def test_existing_integer_decimal_and_fraction_fallbacks_still_work(self):
        cases = [
            ("42", "42"),
            ("-12", "-12"),
            ("3.125", "3.125"),
            ("-3.125", "-3.125"),
            ("7 / 8", "7 / 8"),
            ("+.5", "+.5"),
        ]
        for completion, expected in cases:
            with self.subTest(completion=completion):
                self.assertEqual(parse_answer_robust(completion), expected)
        self.assertEqual(parse_answer_robust("The result is 6%."), "6")


if __name__ == "__main__":
    unittest.main()
