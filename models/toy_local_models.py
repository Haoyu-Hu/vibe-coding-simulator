from __future__ import annotations

import ast
import hashlib
import random
import re
from dataclasses import dataclass


_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Load,
    ast.Call,
    ast.Name,
)

_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


@dataclass
class LocalModelConfig:
    model_id: str
    temperature: float = 0.0
    seed: int = 7


def _stable_noise(model_id: str) -> float:
    digest = hashlib.md5(model_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value


def model_quality(model_id: str) -> float:
    name = model_id.lower()
    if any(x in name for x in ("large", "xl", "70b", "72b", "gpt-5", "o3", "o4")):
        return 0.9
    if any(x in name for x in ("medium", "8b", "14b", "sonnet")):
        return 0.75
    if any(x in name for x in ("small", "mini", "1.5b", "3b", "7b")):
        return 0.6
    return 0.45 + 0.4 * _stable_noise(model_id)


def _safe_eval_expr(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ValueError("Unsupported expression")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Unsupported function")
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _ALLOWED_FUNCS))


def try_compute_math(question: str) -> str | None:
    inline_match = re.search(r"(-?\d+[\d\s\+\-\*\/\(\)\.]{0,80})", question)
    if inline_match:
        expr = inline_match.group(1)
        expr = expr.replace("^", "**")
        expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s]", "", expr)
        expr = re.sub(r"\s+", "", expr)
        if expr:
            try:
                value = _safe_eval_expr(expr)
                if abs(value - round(value)) < 1e-9:
                    return str(int(round(value)))
                return f"{value:.6f}".rstrip("0").rstrip(".")
            except Exception:
                pass

    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", question)]
    q = question.lower()
    if len(numbers) >= 2:
        a, b = numbers[0], numbers[1]
        if "sum" in q or "plus" in q or "add" in q:
            result = a + b
        elif "difference" in q or "minus" in q or "subtract" in q:
            result = a - b
        elif "product" in q or "times" in q or "multiply" in q:
            result = a * b
        elif "divide" in q or "quotient" in q:
            if b == 0:
                return None
            result = a / b
        else:
            return None
        if abs(result - round(result)) < 1e-9:
            return str(int(round(result)))
        return f"{result:.6f}".rstrip("0").rstrip(".")
    return None


def predict_math_answer(
    question: str,
    config: LocalModelConfig,
    turn_index: int = 1,
    max_turns: int = 1,
    improvement: float = 0.0,
) -> str:
    rng = random.Random(config.seed + turn_index * 131)
    base = try_compute_math(question)
    if base is None:
        base = "0"

    quality = model_quality(config.model_id)
    turn_bonus = 0.0
    if max_turns > 1:
        turn_bonus = 0.2 * (turn_index - 1) / max(max_turns - 1, 1)
    p_correct = min(0.99, max(0.05, quality + turn_bonus + improvement))
    if rng.random() < p_correct:
        return base

    # Controlled incorrect answer for stable evaluation.
    try:
        bad = float(base)
    except ValueError:
        return "unknown"
    delta = rng.choice([-2.0, -1.0, 1.0, 2.0])
    wrong = bad + delta
    if abs(wrong - round(wrong)) < 1e-9:
        return str(int(round(wrong)))
    return f"{wrong:.4f}".rstrip("0").rstrip(".")


def _mutate_code(reference: str, rng: random.Random) -> str:
    lines = reference.splitlines()
    if not lines:
        return reference
    mutated = list(lines)
    for i, line in enumerate(mutated):
        stripped = line.strip()
        if " return " in f" {stripped} ":
            if "+" in line:
                mutated[i] = line.replace("+", "-", 1)
                return "\n".join(mutated)
            if "*" in line:
                mutated[i] = line.replace("*", "+", 1)
                return "\n".join(mutated)
            if "-" in line:
                mutated[i] = line.replace("-", "+", 1)
                return "\n".join(mutated)
            if "len(" in line:
                mutated[i] = line.replace("len(", "max(0, len(", 1)
                return "\n".join(mutated)
    # Fallback mutation: append harmless but wrong override for known variables.
    return reference + "\n\n# Mutation\n"


def generate_code_candidate(task: dict, config: LocalModelConfig, quality_boost: float = 0.0) -> str:
    reference = task.get("reference_solution")
    if not reference:
        return "pass"

    rng = random.Random(config.seed + len(task.get("task_id", "")) * 17)
    quality = min(0.99, model_quality(config.model_id) + quality_boost)
    if rng.random() < quality:
        return reference
    return _mutate_code(reference, rng)


def predict_qa_answer(question: str, answer: str, config: LocalModelConfig) -> str:
    rng = random.Random(config.seed + len(question))
    quality = model_quality(config.model_id)
    if rng.random() < quality:
        return answer

    # Produce a plausible but wrong answer by changing one token.
    tokens = answer.split()
    if not tokens:
        return "unknown"
    tokens[-1] = "later"
    return " ".join(tokens)


def generate_dialogue_response(
    last_user_message: str,
    state: dict,
    profile: str,
    config: LocalModelConfig,
    use_state_alignment: bool,
) -> str:
    emotion = str(state.get("emotion", "neutral"))
    goal = str(state.get("goal", "respond"))
    stance = str(state.get("stance", "neutral"))
    belief = str(state.get("belief", ""))

    if use_state_alignment:
        return (
            f"I hear you. Based on your goal ({goal}) and stance ({stance}), "
            f"I suggest this step: {belief}. Tone: {emotion}."
        )

    return f"Thanks for sharing. You said: {last_user_message}. I can help with that."
