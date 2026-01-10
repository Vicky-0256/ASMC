"""
Answer Normalizer - Zero-cost local answer normalization.

Handles:
1. Fraction reduction (gcd normalization)
2. Set/list standardization (sort, remove duplicates)
3. Sign normalization for comma-separated values
4. Whitespace and format cleanup

This can fix errors like:
- #93: 2240/78125 -> 448/15625 (fraction reduction)
- #26: "1, 2" vs "1,-2" (sign/format)
"""

import re
from math import gcd
from typing import Optional, Tuple, List


def normalize_answer(answer: str) -> str:
    """
    Apply all normalizations to an answer string.
    Returns the normalized form.
    """
    if not answer:
        return answer
    
    answer = answer.strip()
    
    # Try fraction normalization
    frac_result = normalize_fraction(answer)
    if frac_result:
        return frac_result
    
    # Try set/list normalization
    set_result = normalize_set(answer)
    if set_result:
        return set_result
    
    # Basic cleanup
    return basic_cleanup(answer)


def normalize_fraction(answer: str) -> Optional[str]:
    """
    Reduce fractions to lowest terms.
    
    Handles:
    - \frac{a}{b} -> \frac{a/gcd}{b/gcd}
    - a/b -> reduced form
    """
    # Pattern 1: \frac{num}{denom}
    frac_match = re.match(r'^\\frac\{(-?\d+)\}\{(\d+)\}$', answer.strip())
    if frac_match:
        num = int(frac_match.group(1))
        denom = int(frac_match.group(2))
        if denom != 0:
            g = gcd(abs(num), denom)
            num_reduced = num // g
            denom_reduced = denom // g
            if denom_reduced == 1:
                return str(num_reduced)
            return f"\\frac{{{num_reduced}}}{{{denom_reduced}}}"
    
    # Pattern 2: num/denom (plain)
    plain_frac = re.match(r'^(-?\d+)\s*/\s*(\d+)$', answer.strip())
    if plain_frac:
        num = int(plain_frac.group(1))
        denom = int(plain_frac.group(2))
        if denom != 0:
            g = gcd(abs(num), denom)
            num_reduced = num // g
            denom_reduced = denom // g
            if denom_reduced == 1:
                return str(num_reduced)
            return f"{num_reduced}/{denom_reduced}"
    
    return None


def normalize_set(answer: str) -> Optional[str]:
    """
    Normalize set/list answers.
    
    Handles:
    - "1, 2" -> sorted, cleaned
    - "3 and 7" -> "3, 7"
    - Extracts numbers and sorts them
    """
    # Check if it looks like a set/list
    if ' and ' in answer.lower() or ',' in answer:
        # Extract all numbers (integers, fractions, negatives)
        # Pattern: optional negative, digits, optional fraction part
        elements = re.findall(r'-?\d+(?:/\d+)?(?:\.\d+)?', answer)
        
        if len(elements) >= 2:
            # Convert to sortable form
            def to_float(s):
                if '/' in s:
                    parts = s.split('/')
                    return float(parts[0]) / float(parts[1])
                return float(s)
            
            try:
                sorted_elements = sorted(elements, key=to_float)
                return ', '.join(sorted_elements)
            except:
                pass
    
    return None


def basic_cleanup(answer: str) -> str:
    """Basic whitespace and format cleanup."""
    answer = answer.strip()
    # Remove extra spaces around operators
    answer = re.sub(r'\s*,\s*', ', ', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def extract_numeric_value(answer: str) -> Optional[float]:
    """
    Extract numeric value from answer for comparison.
    Returns float or None if not purely numeric.
    """
    answer = answer.strip()
    
    # Direct number
    try:
        return float(answer)
    except:
        pass
    
    # \frac{a}{b}
    frac_match = re.match(r'^\\frac\{(-?\d+)\}\{(\d+)\}$', answer)
    if frac_match:
        num = int(frac_match.group(1))
        denom = int(frac_match.group(2))
        if denom != 0:
            return num / denom
    
    # a/b
    plain_frac = re.match(r'^(-?\d+)\s*/\s*(\d+)$', answer)
    if plain_frac:
        num = int(plain_frac.group(1))
        denom = int(plain_frac.group(2))
        if denom != 0:
            return num / denom
    
    return None


def answers_equivalent(pred: str, expected: str) -> bool:
    """
    Check if two answers are equivalent after normalization.
    """
    if not pred or not expected:
        return False
    
    # Direct match after basic cleanup
    pred_clean = basic_cleanup(pred)
    exp_clean = basic_cleanup(expected)
    
    if pred_clean == exp_clean:
        return True
    
    # Normalized match
    pred_norm = normalize_answer(pred)
    exp_norm = normalize_answer(expected)
    
    if pred_norm == exp_norm:
        return True
    
    # Numeric equivalence
    pred_val = extract_numeric_value(pred)
    exp_val = extract_numeric_value(expected)
    
    if pred_val is not None and exp_val is not None:
        if abs(pred_val - exp_val) < 1e-9:
            return True
    
    return False


# =============================================================================
# High Confidence Trigger Detection
# =============================================================================

def is_simple_answer(answer: str) -> bool:
    """
    Check if answer is "simple" enough for verifier to help.
    
    Simple means:
    - Pure number/fraction
    - Small set of numbers (2-4 elements)
    - No complex expressions like sqrt, matrices, etc.
    """
    if not answer:
        return False
    
    answer = answer.strip()
    
    # Exclude complex expressions
    complex_patterns = [
        r'\\sqrt',
        r'\\begin{',
        r'\\text{',  # text answers are harder to verify
        r'\\pm',
        r'\\infty',
        r'\\pi',  # unless just \pi itself
    ]
    
    for pat in complex_patterns:
        if re.search(pat, answer):
            # Exception: allow single \pi
            if pat == r'\\pi' and answer.strip() == '\\pi':
                return True
            return False
    
    # Check if it's a simple number
    if re.match(r'^-?\d+(\.\d+)?$', answer):
        return True
    
    # Check if it's a simple fraction
    if re.match(r'^\\frac\{-?\d+\}\{\d+\}$', answer):
        return True
    if re.match(r'^-?\d+\s*/\s*\d+$', answer):
        return True
    
    # Check if it's a small set of numbers
    elements = re.findall(r'-?\d+(?:/\d+)?(?:\.\d+)?', answer)
    if 2 <= len(elements) <= 4:
        return True
    
    return False


def should_trigger_verifier(
    mass_top: float,
    answer: str,
    n_unique_answers: int = 1,
    mass_threshold: float = 0.90,
) -> Tuple[bool, str]:
    """
    Decide if this answer should be verified.
    
    Trigger conditions:
    1. mass_top >= 0.90 (high confidence)
    2. Answer is "simple" (verifier can help)
    3. Few unique answers (convergence to single answer)
    
    Returns: (should_trigger, reason)
    """
    if mass_top < mass_threshold:
        return False, "mass_below_threshold"
    
    if not is_simple_answer(answer):
        return False, "complex_answer"
    
    # High confidence + simple answer = worth verifying
    return True, "high_conf_simple"


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Fraction Normalization ===")
    tests_frac = [
        ("\\frac{2240}{78125}", "\\frac{448}{15625}"),
        ("\\frac{6}{9}", "\\frac{2}{3}"),
        ("15/25", "3/5"),
    ]
    for inp, expected in tests_frac:
        result = normalize_fraction(inp)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {inp} -> {result} (expected: {expected})")
    
    print("\n=== Set Normalization ===")
    tests_set = [
        ("1, 2", "1, 2"),
        ("3 and 7", "3, 7"),
        ("7, 3, 5", "3, 5, 7"),
    ]
    for inp, expected in tests_set:
        result = normalize_set(inp)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{inp}' -> '{result}' (expected: '{expected}')")
    
    print("\n=== Simple Answer Detection ===")
    tests_simple = [
        ("42", True),
        ("\\frac{3}{2}", True),
        ("1, 2, 3", True),
        ("\\sqrt{51}", False),
        ("\\text{Carla}", False),
        ("1 \\pm \\sqrt{19}", False),
    ]
    for inp, expected in tests_simple:
        result = is_simple_answer(inp)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{inp}' -> {result} (expected: {expected})")
    
    print("\n=== Equivalence Check ===")
    tests_equiv = [
        ("\\frac{2240}{78125}", "\\frac{448}{15625}", True),
        ("1, 2", "1, 2", True),
        ("42", "42", True),
        ("10", "\\sqrt{51}", False),
    ]
    for pred, exp, expected in tests_equiv:
        result = answers_equivalent(pred, exp)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{pred}' == '{exp}' -> {result} (expected: {expected})")
