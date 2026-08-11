from .parse_utils import parse_answer
from .math_grader import grade_answer
from .math_normalize import normalize_answer

# Backward-compatible public name used by earlier experiment code.  The
# bundled Hendrycks MATH normalizer exports ``normalize_answer``; it has never
# defined a separate ``normalize_final_answer`` implementation.
normalize_final_answer = normalize_answer

__all__ = [
    "grade_answer",
    "normalize_answer",
    "normalize_final_answer",
    "parse_answer",
]
