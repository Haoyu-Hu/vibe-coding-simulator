from __future__ import annotations

import re
from fractions import Fraction


def normalize_answer(answer: str) -> str:
    text = answer.strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_final_answer(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    for prefix in ("answer:", "final:", "output:"):
        if last.lower().startswith(prefix):
            return last[len(prefix) :].strip()

    # Accept values like "2 + 3 = 5" by taking the last numeric token.
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", last)
    if matches:
        return matches[-1]
    return last


def _to_fraction(value: str) -> Fraction | None:
    value = value.strip()
    if not value:
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(prediction: str, gold: str, tol: float = 1e-6) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True

    pred_frac = _to_fraction(pred_norm)
    gold_frac = _to_fraction(gold_norm)
    if pred_frac is not None and gold_frac is not None:
        try:
            tol_frac = Fraction(str(tol))
        except (ValueError, ZeroDivisionError):
            tol_frac = Fraction(0)
        return abs(pred_frac - gold_frac) <= tol_frac
    return False
