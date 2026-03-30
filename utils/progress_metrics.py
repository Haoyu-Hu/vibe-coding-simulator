from __future__ import annotations

import ast
import re
import warnings
from collections import Counter
from difflib import SequenceMatcher
from fractions import Fraction
from typing import Any

from utils.eval_code import extract_python_code
from utils.eval_math import extract_final_answer, is_correct, normalize_answer
from utils.eval_text import normalize_text

# Metric inspirations:
# - CodeBLEU-style code similarity (syntax + semantics) for code generation progress.
# - Process supervision style step alignment for math reasoning progress.
# - SQuAD-style token F1 for semantic-answer overlap.


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _token_overlap(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _split_reasoning_steps(text: str) -> list[str]:
    raw_parts: list[str] = []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        for part in re.split(r"(?<=[\.;])\s+|^\s*\d+[\).\s]+", value):
            cleaned = part.strip()
            if cleaned:
                raw_parts.append(cleaned)
    steps = [x for x in raw_parts if len(_token_set(x)) >= 2]
    return steps[:80]


def _math_symbol_set(text: str) -> set[str]:
    patterns = [
        r"\\[a-zA-Z]+(?:\{[^{}]*\})*",
        r"-?\d+(?:\.\d+)?(?:/\d+)?",
        r"[a-zA-Z]+(?:_[a-zA-Z0-9]+)?",
    ]
    symbols: set[str] = set()
    for pat in patterns:
        for token in re.findall(pat, text):
            token = token.strip().lower()
            if token:
                symbols.add(token)
    return symbols


def _safe_fraction(value: str) -> Fraction | None:
    norm = normalize_answer(value)
    if not norm:
        return None
    try:
        return Fraction(norm)
    except Exception:
        return None


def _numeric_closeness(pred: str, gold: str) -> float:
    pred_frac = _safe_fraction(pred)
    gold_frac = _safe_fraction(gold)
    if pred_frac is None or gold_frac is None:
        return 0.0
    dist = abs(pred_frac - gold_frac)
    # Keep computation in rational space to avoid overflow on huge numerators.
    try:
        return float(Fraction(1, 1) / (Fraction(1, 1) + dist))
    except OverflowError:
        return 0.0


def _numeric_difference(pred: str, gold: str) -> float | None:
    pred_frac = _safe_fraction(pred)
    gold_frac = _safe_fraction(gold)
    if pred_frac is None or gold_frac is None:
        return None
    # Store squared error per sample so round-level mean is MSE.
    dist = abs(pred_frac - gold_frac)
    sq_err = dist * dist
    try:
        return float(sq_err)
    except OverflowError:
        # Cap extremely large numeric gaps so metrics remain finite and serializable.
        return 1e308


def math_progress_score(
    *,
    answer: str,
    thinking: str,
    gold_answer: str,
    canonical_solution: str | None,
) -> tuple[float, dict[str, float | None]]:
    pred_final = extract_final_answer(answer)
    gold_final = extract_final_answer(gold_answer)
    final_correct = 1.0 if is_correct(pred_final, gold_final) else 0.0
    numerical_difference = _numeric_difference(pred_final, gold_final)

    # Fallback mode when no worked solution is available.
    if not canonical_solution:
        numeric_progress = _numeric_closeness(pred_final, gold_final)
        token_progress = _token_overlap(answer, gold_answer)
        score = _clamp(0.60 * final_correct + 0.25 * numeric_progress + 0.15 * token_progress)
        return score, {
            "final_correct": final_correct,
            "numeric_closeness": numeric_progress,
            "numerical_difference": numerical_difference,
            "token_overlap": token_progress,
        }

    reference_text = canonical_solution.strip()
    candidate_text = (thinking.strip() + "\n" + answer.strip()).strip()
    ref_steps = _split_reasoning_steps(reference_text)
    cand_steps = _split_reasoning_steps(candidate_text)

    if ref_steps and cand_steps:
        best_scores: list[float] = []
        for ref in ref_steps:
            best = max((_token_overlap(ref, cand) for cand in cand_steps), default=0.0)
            best_scores.append(best)
        step_coverage = sum(best_scores) / len(best_scores)
    else:
        step_coverage = 0.0

    ref_symbols = _math_symbol_set(reference_text)
    cand_symbols = _math_symbol_set(candidate_text)
    if ref_symbols and cand_symbols:
        symbol_overlap = len(ref_symbols & cand_symbols) / len(ref_symbols | cand_symbols)
    else:
        symbol_overlap = 0.0

    score = _clamp(0.55 * step_coverage + 0.30 * symbol_overlap + 0.15 * final_correct)
    return score, {
        "step_coverage": step_coverage,
        "symbol_overlap": symbol_overlap,
        "final_correct": final_correct,
        "numerical_difference": numerical_difference,
    }


def _node_type_counter(code: str) -> Counter[str]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            tree = ast.parse(code)
    except Exception:
        return Counter()
    return Counter(type(node).__name__ for node in ast.walk(tree))


def _counter_jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    if union <= 0:
        return 0.0
    return inter / union


def _identifier_set(code: str) -> set[str]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            tree = ast.parse(code)
    except Exception:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _set_f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def code_progress_score(answer: str, reference_code: str) -> tuple[float, dict[str, float | None]]:
    pred_code = extract_python_code(answer)
    gold_code = extract_python_code(reference_code)

    if not pred_code.strip() or not gold_code.strip():
        return 0.0, {
            "lexical_similarity": 0.0,
            "ast_similarity": 0.0,
            "identifier_f1": 0.0,
        }

    lexical = SequenceMatcher(None, pred_code, gold_code).ratio()
    ast_similarity = _counter_jaccard(_node_type_counter(pred_code), _node_type_counter(gold_code))
    identifier_f1 = _set_f1(_identifier_set(pred_code), _identifier_set(gold_code))
    score = _clamp(0.40 * lexical + 0.40 * ast_similarity + 0.20 * identifier_f1)
    return score, {
        "lexical_similarity": lexical,
        "ast_similarity": ast_similarity,
        "identifier_f1": identifier_f1,
    }


def semantic_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    overlap = sum(common.values())
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def progress_for_sample(
    *,
    sample: dict[str, Any],
    answer: str,
    thinking: str,
    correct: bool,
    eval_details: dict[str, Any],
) -> tuple[float, str, dict[str, float | None]]:
    task_type = sample.get("task_type", "")

    if task_type == "math":
        metadata = sample.get("metadata", {}) or {}
        canonical = metadata.get("canonical_solution")
        score, components = math_progress_score(
            answer=answer,
            thinking=thinking,
            gold_answer=str(sample.get("reference_answer", "")),
            canonical_solution=str(canonical) if canonical is not None else None,
        )
        return score, "math_step_progress", components

    if task_type == "code":
        score, components = code_progress_score(answer, str(sample.get("reference_answer", "")))
        return score, "codebleu_proxy", components

    if task_type == "swebench":
        overlap = float(eval_details.get("changed_file_overlap", 0.0) or 0.0)
        sim = float(eval_details.get("patch_similarity", 0.0) or 0.0)
        score = _clamp(max(1.0 if correct else 0.0, 0.50 * overlap + 0.50 * sim))
        return score, "swebench_patch_progress", {
            "changed_file_overlap": overlap,
            "patch_similarity": sim,
        }

    score = semantic_token_f1(answer, str(sample.get("reference_answer", "")))
    return score, "semantic_token_f1", {"token_f1": score}
