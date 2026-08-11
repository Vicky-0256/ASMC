"""
Answer parsing utilities for math problem solutions.

Supports multiple extraction strategies with priority scoring:
1. \boxed{...} / \fbox{...} (highest priority)
2. Final Answer: / Answer: lines
3. ```output ... ``` code blocks (Qwen-Math style)
4. Print output lines
5. Inline math $...$
6. "is X" patterns
7. Fallback standalone numbers
"""

import json
import re
import numpy as np
from typing import Optional, List, Tuple


_SIGNED_DECIMAL = r"[+-]?(?:\d+\.\d+|\.\d+)"
_SIGNED_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


def _normalize(ans: str) -> str:
    """
    Normalize an answer string:
    - Strip whitespace
    - Remove surrounding $...$
    - Handle "n = 6" -> "6"
    - Remove \left \right
    """
    ans = (ans or "").strip()
    if not ans:
        return ""
    
    # Strip surrounding $...$
    if len(ans) >= 2 and ans[0] == "$" and ans[-1] == "$":
        ans = ans[1:-1].strip()
    
    # Remove \left \right
    ans = ans.replace("\\left", "").replace("\\right", "")
    
    # Collapse whitespace
    ans = re.sub(r"\s+", " ", ans).strip()
    
    # Handle "n = 6" or "x = 42" -> take RHS
    m = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$", ans)
    if m:
        ans = m.group(1).strip()
    
    # Strip sentence-ending periods without deleting the decimal point from a
    # conventional leading-decimal answer such as ``.35625``.
    ans = ans.rstrip(".").strip()
    
    return ans


def remove_boxed(s):
    """Extract content from \boxed{...} string."""
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)


def last_boxed_only_string(string):
    """Find the last \boxed{...} or \fbox{...} in string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def _extract_output_blocks(text: str) -> List[str]:
    """
    Extract ```output ... ``` fenced code blocks.
    
    Qwen-Math often outputs results like:
    ```output
    6
    ```
    
    Returns list of block contents (may have multiple).
    """
    blocks = []
    # Match ```output (with optional whitespace/newline variations)
    pattern = r"```\s*output\s*\n(.*?)\n\s*```"
    for m in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1))
    return blocks


def _extract_print_outputs(text: str) -> List[str]:
    """
    Heuristic: detect standalone number lines that appear to be print outputs.
    
    Only triggers if there's a print(...) call somewhere in recent text.
    Returns standalone numeric values (not assignments like n = 1).
    """
    # Focus on last ~3000 chars
    tail = text[-3000:] if len(text) > 3000 else text
    lines = tail.splitlines()
    
    # Check if there's a print() call in this region
    has_print = any(re.search(r"\bprint\s*\(", ln) for ln in lines)
    if not has_print:
        return []
    
    outputs = []
    # Look for standalone numbers (not part of assignment)
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # Skip if it looks like an assignment (n = 1, x = 42)
        if re.match(r"^[a-zA-Z_]\w*\s*=", s):
            continue
        # Skip if it's inside a code block marker
        if s.startswith("```") or s.startswith("#"):
            continue
        # Match standalone integer
        if re.match(r"^[+-]?\d+$", s):
            outputs.append(s)
        # Match standalone decimal
        elif re.match(rf"^{_SIGNED_DECIMAL}$", s):
            outputs.append(s)
        # Match standalone fraction
        elif re.match(r"^[+-]?\d+\s*/\s*\d+$", s):
            outputs.append(s)
    
    return outputs


def _extract_final_answer_lines(text: str) -> List[str]:
    """
    Extract answers from "Final Answer: X" or "Answer: X" patterns.
    """
    patterns = [
        r"Final\s*Answer\s*[:：]\s*(.+)",
        r"Answer\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
        r"Therefore,?\s+the\s+answer\s+is\s*[:=]?\s*(.+)",
    ]
    
    results = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            # Take first line only
            ans = m.group(1).splitlines()[0].strip()
            if ans:
                results.append(ans)
    return results


def parse_answer(input_str):
    """Original simple parser - just extracts \boxed{...}."""
    return remove_boxed(last_boxed_only_string(input_str))


def parse_answer_robust(input_str: str, return_source: bool = False, return_all: bool = False):
    """
    Robust answer extraction with candidate scoring.
    
    Priority (score):
    1. \boxed{...} / \fbox{...}     (100)
    2. Final Answer: lines           (95)
    3. ```output ... ``` blocks      (80)  <- Reduced from 85, needs verification
    4. Print output lines            (75)
    5. Inline math $...$             (65)
    6. "is X" patterns               (55)
    7. Fallback standalone number    (10)
    
    Args:
        input_str: The completion text to parse
        return_source: If True, returns (answer, source) tuple
        return_all: If True, returns List[(answer, score, source)] of all candidates
        
    Returns:
        - Default: answer string
        - return_source=True: (answer, source) tuple
        - return_all=True: List[(answer, score, source)] sorted by score desc
    """
    if input_str is None or not isinstance(input_str, str):
        if return_all:
            return []
        if return_source:
            return (None, None)
        return None
    
    text = input_str
    # Focus on tail for most patterns (except boxed which searches full text)
    tail = text[-4000:] if len(text) > 4000 else text
    
    candidates: List[Tuple[int, str, str]] = []  # (score, answer, source)
    
    # =================================================================
    # Strategy 1: \boxed{...} or \fbox{...}  (score=100)
    # =================================================================
    boxed_str = last_boxed_only_string(text)
    if boxed_str:
        inner = remove_boxed(boxed_str)
        if inner:
            candidates.append((100, _normalize(inner), "boxed"))
    
    # =================================================================
    # Strategy 2: Final Answer: / Answer: lines  (score=95)
    # =================================================================
    final_answers = _extract_final_answer_lines(tail)
    for ans in final_answers:
        norm = _normalize(ans)
        if norm:
            candidates.append((95, norm, "final_line"))
    
    # =================================================================
    # Strategy 3: ```output ... ``` blocks  (score=80)
    # Reduced from 85 - needs independent verification
    # =================================================================
    output_blocks = _extract_output_blocks(tail)
    if output_blocks:
        # Take last output block, get last non-empty line
        last_block = output_blocks[-1]
        lines = [ln.strip() for ln in last_block.splitlines() if ln.strip()]
        if lines:
            # Take last line of output block
            norm = _normalize(lines[-1])
            if norm:
                candidates.append((80, norm, "output_block"))
    
    # =================================================================
    # Strategy 4: Print outputs (non-fenced)  (score=75)
    # =================================================================
    print_outputs = _extract_print_outputs(tail)
    if print_outputs:
        # Take last print output
        norm = _normalize(print_outputs[-1])
        if norm:
            candidates.append((75, norm, "print_output"))
    
    # =================================================================
    # Strategy 5: Last inline math $...$  (score=65)
    # =================================================================
    # Look for $...$ but avoid matching inside code blocks
    # Remove code blocks first for this search
    text_no_code = re.sub(r"```.*?```", "", tail, flags=re.DOTALL)
    math_matches = list(re.finditer(r"\$([^$]{1,200})\$", text_no_code))
    if math_matches:
        last_math = math_matches[-1].group(1)
        norm = _normalize(last_math)
        if norm:
            candidates.append((65, norm, "inline_math"))
    
    # =================================================================
    # Strategy 6: "is X" patterns near end  (score=55)
    # e.g., "the answer is 6", "is $n = 6$"
    # =================================================================
    is_patterns = [
        r"\bis\s+\$([^$]+)\$",           # is $...$
        r"\bis\s+\\boxed\{([^}]+)\}",    # is \boxed{...}
        # is 6, is 6.5, or is .5; retain the prior non-word end boundary.
        rf"\bis\s+({_SIGNED_NUMBER})(?!\w)",
        r"=\s*\$([^$]+)\$\s*[.\n]",      # = $...$.
    ]
    
    for pat in is_patterns:
        matches = list(re.finditer(pat, tail[-1000:], flags=re.IGNORECASE))
        if matches:
            last_match = matches[-1].group(1)
            norm = _normalize(last_match)
            if norm:
                candidates.append((55, norm, "is_pattern"))
                break  # Take first matching pattern type
    
    # =================================================================
    # Strategy 7: Fallback - last standalone number line  (score=10)
    # Very aggressive, only if nothing else works
    # =================================================================
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    for ln in reversed(lines):
        # Skip code block markers
        if ln.startswith("```") or ln.startswith("#"):
            continue
        # Skip assignments
        if re.match(r"^[a-zA-Z_]\w*\s*=", ln):
            continue
        # Match standalone number
        if re.match(r"^[+-]?\d+$", ln):
            candidates.append((10, ln, "fallback_number"))
            break
        if re.match(rf"^{_SIGNED_DECIMAL}$", ln):
            candidates.append((10, ln, "fallback_number"))
            break
        if re.match(r"^[+-]?\d+\s*/\s*\d+$", ln):
            candidates.append((10, ln, "fallback_number"))
            break
    
    # =================================================================
    # Select best candidate or return all
    # =================================================================
    # Filter empty answers
    candidates = [(s, a, src) for (s, a, src) in candidates if a]
    
    if not candidates:
        if return_all:
            return []
        if return_source:
            return (None, None)
        return None
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return all candidates if requested
    if return_all:
        # Return as List[(answer, score, source)]
        return [(a, s, src) for (s, a, src) in candidates]
    
    best_score, best_answer, best_source = candidates[0]
    
    if return_source:
        return (best_answer, best_source)
    return best_answer
