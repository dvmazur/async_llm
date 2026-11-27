"""Examples from HuggingFaceH4/MATH-500 with equivalent and incorrect predictions."""

import pytest
from minisgl_examples.training.parallel_thinking.math500 import grade_math500


@pytest.mark.parametrize(
    ("gold", "prediction", "grade"),
    [
        # The official first test row asks for ordered polar coordinates.
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3, 0.5\pi)", "correct"),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(\pi/2, 3)", "incorrect"),
        ("p - q", "-q + p", "correct"),
        (r"\frac{14}{3}", r"\dfrac{28}{6}", "correct"),
        ("(6,31,-1)", "(-1,31,6)", "incorrect"),
        ("1,-2", "-2,1", "correct"),  # Unordered roots.
        ("(3,4]", "[3,4]", "incorrect"),
        (r"\{1\pm\sqrt{5},-2\}", r"\{-2,1-\sqrt5,1+\sqrt5\}", "correct"),
        (r"\text{Evelyn}", "Evelyn", "correct"),
        (r"\text{east}", "SEAT", "incorrect"),
        (r"\text{(C)}", "C", "correct"),
        (r"\frac{14}{3}", r"\frac{", "unparsed_prediction"),
        ("9", None, "unparsed_prediction"),
        (r"\frac{", "9", "unparsed_gold"),
    ],
)
def test_math500_grade(gold, prediction, grade):
    assert grade_math500(gold, prediction) == grade
