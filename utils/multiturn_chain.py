from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from models.toy_local_models import (
    LocalModelConfig,
    generate_code_candidate,
    model_quality,
    predict_math_answer,
    predict_qa_answer,
)
from utils.eval_code import build_candidate, extract_python_code, run_tests
from utils.eval_math import extract_final_answer, is_correct
from utils.eval_text import exact_match
from utils.metrics import trapezoid_auc
from utils.progress_metrics import progress_for_sample, semantic_token_f1

PIPELINE_COMBINATIONS = [
    {"output_mode": "answer_only", "context_mode": "last_round"},
    {"output_mode": "answer_only", "context_mode": "full_history"},
    {"output_mode": "answer_with_thinking", "context_mode": "last_round"},
    {"output_mode": "answer_with_thinking", "context_mode": "full_history"},
]

LOCAL_MODEL_MAPS = {
    "qwen": {
        "math": "Qwen/Qwen2.5-Math-7B-Instruct",
        "code": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "common": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama": {
        "math": "meta-llama/Llama-3.1-8B-Instruct",
        "code": "meta-llama/CodeLlama-7b-Instruct-hf",
        "common": "meta-llama/Llama-3.2-3B-Instruct",
    },
}

API_MODEL_MAP = {
    "math": "openai/gpt-4o-mini",
    "code": "anthropic/claude-3.5-haiku",
    "common": "google/gemini-2.0-flash-001",
}

API_CLOSED_MODELS = [
    # "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
]

API_OPEN_MODEL_MAPS = {
    "qwen": {
        "math": "qwen/qwen-2.5-7b-instruct",
        "code": "qwen/qwen-2.5-coder-32b-instruct",
        "common": "qwen/qwen-2.5-7b-instruct",
    },
    "llama": {
        "math": "meta-llama/llama-3.1-8b-instruct",
        "code": "meta-llama/llama-3.1-8b-instruct",
        "common": "meta-llama/llama-3.2-3b-instruct",
    },
}

API_MODELS = [
    # "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
]

TRACE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
NATURALPLAN_SEMANTIC_CORRECT_THRESHOLD = 0.50

THINKING_PIPELINE_COMBINATIONS = [
    {"output_mode": "answer_with_thinking", "context_mode": "last_round"},
    {"output_mode": "answer_with_thinking", "context_mode": "full_history"},
]

ANSWER_ONLY_PIPELINE_COMBINATIONS = [
    {"output_mode": "answer_only", "context_mode": "last_round"},
    {"output_mode": "answer_only", "context_mode": "full_history"},
]

LAST_ROUND_PIPELINE_COMBINATIONS = [
    {"output_mode": "answer_only", "context_mode": "last_round"},
    {"output_mode": "answer_with_thinking", "context_mode": "last_round"},
]


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def condition_name(pipeline: dict[str, str]) -> str:
    return f"{pipeline['output_mode']}__{pipeline['context_mode']}"


def sample_question_id(sample_id: str) -> str:
    if "::" in sample_id:
        return sample_id.split("::", 1)[1]
    return sample_id


def data_file_name(sample_id: str) -> str:
    qid = slugify(sample_question_id(sample_id)).replace("-", "_")
    return f"data_{qid}.jsonl"


def canonical_dataset_name(name: str) -> str:
    return re.sub(r"-(mini|full)$", "", name.strip().lower())


def is_swebench_dataset(name: str) -> bool:
    return canonical_dataset_name(name) == "swebench"


def _normalize_reference_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    # NaturalPlan references may be serialized as a JSON string literal
    # (wrapped in quotes with escaped newlines). Unwrap up to twice.
    for _ in range(2):
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            try:
                decoded = json.loads(text)
            except Exception:
                break
            if isinstance(decoded, str):
                text = decoded.strip()
                continue
        break

    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def condition_dir(exp_dir: Path, chain: dict[str, Any]) -> Path:
    return (
        exp_dir
        / "chain"
        / slugify(chain["dataset_name"])
        / slugify(chain["model_name"])
        / slugify(condition_name(chain["pipeline"]))
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-zA-Z0-9]+", a.lower()))
    b_tokens = set(re.findall(r"[a-zA-Z0-9]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def compute_cot_monitor_score(question: str, answer: str, thinking: str, correct: bool) -> float:
    score = 0.15 + (0.35 if correct else 0.0)
    if thinking:
        words = len(thinking.split())
        score += min(0.25, words / 120.0)
        score += 0.25 * _token_overlap(question, thinking)
    else:
        score += 0.05
    if not answer.strip():
        score -= 0.2
    return clamp(score)


def compute_trace_score(answer: str, thinking: str, correct: bool) -> float:
    if not thinking.strip():
        return 0.0
    ys: list[float] = []
    for frac in TRACE_FRACTIONS:
        end = max(1, int(len(thinking) * frac))
        partial = thinking[:end]
        consistency = text_similarity(partial[-120:], thinking[-120:])
        anchor = text_similarity(answer.lower(), partial.lower())
        reward = 0.25 * frac + 0.35 * consistency + 0.25 * anchor + (0.15 if correct else 0.0)
        ys.append(clamp(reward))
    return trapezoid_auc(TRACE_FRACTIONS, ys)


def _extract_answer_artifact(raw_text: str, task_type: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    if task_type == "code":
        return extract_python_code(text)
    if task_type == "swebench":
        return _extract_unified_diff(text)
    return text


def parse_model_output(raw_response: str, output_mode: str, task_type: str = "text") -> tuple[str, str]:
    text = raw_response.strip()
    thinking = ""
    answer = ""

    if output_mode == "answer_with_thinking":
        think_match = re.search(r"reasoning\s*:\s*(.*?)(?:\n\s*answer\s*:|\Z)", text, re.IGNORECASE | re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        answer_match = re.search(r"answer\s*:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        if not answer:
            answer = _extract_answer_artifact(text, task_type)
        if task_type in {"code", "swebench"} and answer:
            answer = _extract_answer_artifact(answer, task_type)
        return thinking, answer

    answer_match = re.search(r"answer\s*:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = _extract_answer_artifact(text, task_type)
    if task_type in {"code", "swebench"} and answer:
        answer = _extract_answer_artifact(answer, task_type)
    return "", answer


def evaluate_answer(
    sample: dict[str, Any],
    answer: str,
    test_timeout: int,
    *,
    thinking: str = "",
) -> tuple[bool, dict[str, Any]]:
    task_type = sample["task_type"]
    details: dict[str, Any] = {"task_type": task_type}

    if task_type == "math":
        parsed = extract_final_answer(answer)
        gold_raw = str(sample.get("reference_answer", ""))
        gold_final = extract_final_answer(gold_raw)
        correct = is_correct(parsed, gold_final)
        details.update(
            {
                "parsed_answer": parsed,
                "gold_answer": gold_raw,
                "gold_final_answer": gold_final,
            }
        )
        progress_score, progress_metric, progress_components = progress_for_sample(
            sample=sample,
            answer=answer,
            thinking=thinking,
            correct=correct,
            eval_details=details,
        )
        details.update(
            {
                "progress_score": progress_score,
                "progress_metric": progress_metric,
                "progress_components": progress_components,
                "numerical_difference": progress_components.get("numerical_difference"),
            }
        )
        return correct, details

    if task_type == "code":
        metadata = sample.get("metadata", {})
        prompt = metadata.get("prompt", sample["question"])
        entry_point = metadata.get("entry_point", "solution")
        test_code = metadata.get("test", "")
        candidate = build_candidate(prompt, answer, entry_point)
        result = run_tests(candidate, test_code, entry_point, timeout=test_timeout)
        details.update(
            {
                "entry_point": entry_point,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:1000],
            }
        )
        progress_score, progress_metric, progress_components = progress_for_sample(
            sample=sample,
            answer=answer,
            thinking=thinking,
            correct=result.passed,
            eval_details=details,
        )
        details.update(
            {
                "progress_score": progress_score,
                "progress_metric": progress_metric,
                "progress_components": progress_components,
            }
        )
        return result.passed, details

    if task_type == "swebench":
        gold_raw = str(sample.get("reference_answer", ""))
        pred_raw = str(answer)
        gold_patch = _normalize_patch(_extract_unified_diff(gold_raw))
        pred_patch = _normalize_patch(_extract_unified_diff(pred_raw))
        gold_files = _patch_changed_files(gold_patch)
        pred_files = _patch_changed_files(pred_patch)
        file_overlap = (
            len(gold_files & pred_files) / len(gold_files)
            if gold_files
            else 0.0
        )
        similarity = text_similarity(pred_patch, gold_patch)
        exact_patch_match = pred_patch == gold_patch and bool(pred_patch)

        # SWE-bench has many valid patches per issue; exact string match is too strict.
        # Use a structured proxy that checks file targeting + patch-content similarity.
        proxy_resolved = bool(pred_patch) and (
            exact_patch_match
            or (file_overlap >= 0.5 and similarity >= 0.15)
            or similarity >= 0.35
        )
        correct = proxy_resolved
        details.update(
            {
                "gold_patch_chars": len(gold_patch),
                "pred_patch_chars": len(pred_patch),
                "gold_changed_files": sorted(gold_files),
                "pred_changed_files": sorted(pred_files),
                "changed_file_overlap": file_overlap,
                "patch_similarity": similarity,
                "exact_patch_match": exact_patch_match,
                "resolution_proxy_resolved": proxy_resolved,
            }
        )
        progress_score, progress_metric, progress_components = progress_for_sample(
            sample=sample,
            answer=answer,
            thinking=thinking,
            correct=correct,
            eval_details=details,
        )
        details.update(
            {
                "progress_score": progress_score,
                "progress_metric": progress_metric,
                "progress_components": progress_components,
            }
        )
        return correct, details

    dataset_base = canonical_dataset_name(str(sample.get("dataset_name", "")))
    gold_raw = str(sample.get("reference_answer", ""))
    gold = _normalize_reference_text(gold_raw) if dataset_base == "naturalplan" else gold_raw
    correct = exact_match(answer, gold)

    # Accept option-label outputs for multiple-choice style text tasks
    # (e.g., "Option A", "A", "2") by mapping labels back to option text.
    if not correct:
        option_map = _extract_question_options(str(sample.get("question", "")))
        pred_label = _extract_predicted_option_label(answer)
        if pred_label and pred_label in option_map:
            mapped_answer = option_map[pred_label]
            correct = exact_match(mapped_answer, gold)
            details["predicted_option_label"] = pred_label
            details["mapped_answer"] = mapped_answer

    # NaturalPlan is open-ended planning text; exact match is too strict.
    # Mark as correct when semantic overlap reaches a calibrated threshold.
    if dataset_base == "naturalplan":
        token_f1 = semantic_token_f1(answer, gold)
        details["semantic_token_f1"] = token_f1
        details["semantic_correct_threshold"] = NATURALPLAN_SEMANTIC_CORRECT_THRESHOLD
        if token_f1 >= NATURALPLAN_SEMANTIC_CORRECT_THRESHOLD:
            correct = True

    details.update({"gold_answer": gold})
    progress_score, progress_metric, progress_components = progress_for_sample(
        sample=sample,
        answer=answer,
        thinking=thinking,
        correct=correct,
        eval_details=details,
    )
    details.update(
        {
            "progress_score": progress_score,
            "progress_metric": progress_metric,
            "progress_components": progress_components,
        }
    )
    return correct, details


def _extract_question_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw in question.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^Option\s+([A-Za-z0-9]+)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            label = match.group(1).strip().upper()
            text = match.group(2).strip()
            options[label] = text
    return options


def _extract_predicted_option_label(answer: str) -> str | None:
    value = answer.strip()
    if not value:
        return None
    # Typical forms: "Option A", "option 2", "A", "2", "Option A: ..."
    match = re.search(r"\boption\s+([A-Za-z0-9]+)\b", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()

    compact = value.replace(":", " ").strip()
    if re.fullmatch(r"[A-Za-z0-9]+", compact):
        return compact.upper()
    return None


def _normalize_patch(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    trimmed = [line.rstrip() for line in lines]
    return "\n".join(trimmed).strip()


def _extract_unified_diff(text: str) -> str:
    source = text.strip()
    if not source:
        return ""

    candidates: list[str] = []

    fenced = re.findall(r"```(?:diff|patch)?\s*(.*?)```", source, flags=re.IGNORECASE | re.DOTALL)
    for chunk in fenced:
        value = chunk.strip()
        if value:
            candidates.append(value)

    idx = source.find("diff --git ")
    if idx >= 0:
        candidates.append(source[idx:].strip())

    if source.startswith("--- ") and "\n+++ " in source:
        candidates.append(source)

    if not candidates:
        return source

    # Prefer the longest extracted block to avoid truncated partial hunks.
    return max(candidates, key=len)


def _patch_changed_files(diff_text: str) -> set[str]:
    files: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"^diff --git a/(.+?) b/(.+?)$", line.strip())
            if match:
                files.add(match.group(1).strip())
                continue
        if line.startswith("--- a/"):
            files.add(line[6:].strip())
    return {f for f in files if f}


def _history_block(chain: dict[str, Any], include_all: bool) -> str:
    nodes = chain["nodes"]
    if not nodes:
        return ""
    if not include_all:
        node = nodes[-1]
        lines = [
            f"Here is the original question: {chain['question']}",
            f"Here is the answer from others: {node.get('answer', '')}",
        ]
        if node.get("thinking"):
            lines.append(
                "Here is the thinking procedure (if we have) of getting this answer: "
                + node.get("thinking", "")
            )
        lines.append("Think carefully about an improvement based on this.")
        return "\n".join(lines)

    lines = [f"Here is the original question: {chain['question']}", "Here is the full answer history:"]
    for node in nodes:
        lines.append(f"Round {node['round']}: answer={node.get('answer', '')}")
        if node.get("thinking"):
            lines.append(f"Round {node['round']}: thinking={node.get('thinking', '')}")
    lines.append("Think carefully about an improvement based on this.")
    return "\n".join(lines)


def build_prompt(chain: dict[str, Any], round_index: int) -> tuple[str, str, dict[str, Any]]:
    output_mode = chain["pipeline"]["output_mode"]
    context_mode = chain["pipeline"]["context_mode"]

    if round_index == 1:
        user_text = f"Original question: {chain['question']}"
    else:
        user_text = _history_block(chain, include_all=(context_mode == "full_history"))

    if chain["task_type"] == "code":
        task_note = "Return valid Python code as the answer body."
    elif chain["task_type"] == "swebench":
        task_note = "Return only a unified diff patch that resolves the issue."
    elif chain["task_type"] == "math":
        task_note = "Return the final numeric answer."
    else:
        task_note = "Return the final concise answer."

    if output_mode == "answer_only":
        if chain["task_type"] == "code":
            format_note = "Output format (strict): Answer: <python code only, no explanation>"
        elif chain["task_type"] == "swebench":
            format_note = "Output format (strict): Answer: <unified diff patch only, no explanation>"
        else:
            format_note = "Output format: Answer: <your answer>"
    else:
        if chain["task_type"] == "code":
            format_note = (
                "Output format (strict):\n"
                "Reasoning: <at most 2 short lines>\n"
                "Answer:\n```python\n<python code only>\n```"
            )
        elif chain["task_type"] == "swebench":
            format_note = (
                "Output format (strict):\n"
                "Reasoning: <at most 2 short lines>\n"
                "Answer:\n```diff\n<unified diff patch only>\n```"
            )
        else:
            format_note = "Output format: Reasoning: <steps>\nAnswer: <your answer>"

    # Baseline runner intentionally does not use a system prompt to avoid
    # introducing hidden steering effects across models/providers.
    system_prompt = ""
    prompt = "\n\n".join([user_text, task_note, format_note])
    meta = {
        "output_mode": output_mode,
        "context_mode": context_mode,
    }
    return prompt, system_prompt, meta


def _stable_rng(chain_id: str, round_index: int, salt: str) -> random.Random:
    key = f"{chain_id}:{round_index}:{salt}".encode("utf-8")
    seed = int(hashlib.md5(key).hexdigest()[:8], 16)
    return random.Random(seed)


def simulate_local_response(chain: dict[str, Any], round_index: int, max_rounds: int) -> str:
    output_mode = chain["pipeline"]["output_mode"]
    context_mode = chain["pipeline"]["context_mode"]
    question = chain["question"]
    task_type = chain["task_type"]
    model_name = chain["model_name"]

    rng = _stable_rng(chain["chain_id"], round_index, "local")
    config = LocalModelConfig(model_id=model_name, seed=rng.randint(1, 10_000))
    history_boost = 0.04 if context_mode == "full_history" else 0.01
    think_boost = 0.03 if output_mode == "answer_with_thinking" else 0.0
    round_boost = 0.25 * (round_index - 1) / max(max_rounds - 1, 1)
    improvement = history_boost + think_boost + round_boost

    if task_type == "math":
        answer = predict_math_answer(
            question,
            config,
            turn_index=round_index,
            max_turns=max_rounds,
            improvement=improvement,
        )
    elif task_type == "code":
        metadata = chain.get("metadata", {})
        task = {
            "task_id": chain["chain_id"],
            "prompt": metadata.get("prompt", question),
            "entry_point": metadata.get("entry_point", "solution"),
            "reference_solution": chain.get("reference_answer", "pass"),
        }
        answer = generate_code_candidate(task, config, quality_boost=improvement)
    elif task_type == "swebench":
        quality = model_quality(model_name)
        success_prob = min(0.95, 0.15 + 0.55 * quality + 0.30 * improvement)
        if rng.random() < success_prob:
            answer = str(chain.get("reference_answer", ""))
        else:
            answer = (
                "diff --git a/file.py b/file.py\n"
                "index 0000000..1111111 100644\n"
                "--- a/file.py\n"
                "+++ b/file.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-pass\n"
                "+pass\n"
            )
    else:
        answer = predict_qa_answer(question, chain.get("reference_answer", ""), config)

    if output_mode == "answer_only":
        return f"Answer: {answer}"

    thinking = (
        f"Round {round_index} reflection with {model_name}. "
        f"Context mode is {context_mode}; quality target is {model_quality(model_name):.2f}."
    )
    return f"Reasoning: {thinking}\nAnswer: {answer}"


def make_chain_id(sample_id: str, pipeline: dict[str, str], model_key: str) -> str:
    base = (
        f"{sample_id}::{pipeline['output_mode']}::{pipeline['context_mode']}::{model_key}"
    )
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    clean = sample_id.replace("/", "_").replace(":", "_")
    return (
        f"{clean}__{pipeline['output_mode']}__{pipeline['context_mode']}__{model_key}__{digest}"
    )


def build_chains(
    samples: list[dict[str, Any]],
    pipeline_mode: str,
    model_profiles: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if pipeline_mode == "all":
        combinations = PIPELINE_COMBINATIONS
    elif pipeline_mode in {"last_round", "last_round_all"}:
        combinations = LAST_ROUND_PIPELINE_COMBINATIONS
    elif pipeline_mode in {"thinking", "thinking_only"}:
        combinations = THINKING_PIPELINE_COMBINATIONS
    elif pipeline_mode in {"answer_only_all", "answer_only"}:
        combinations = ANSWER_ONLY_PIPELINE_COMBINATIONS
    else:
        parts = pipeline_mode.split("+")
        if len(parts) != 2:
            raise ValueError(
                "Invalid pipeline mode. Use all, last_round, thinking, thinking_only,"
                " answer_only, answer_only_all, or <output_mode>+<context_mode>."
            )
        if parts[0] not in {"answer_only", "answer_with_thinking"}:
            raise ValueError("Invalid output_mode. Use answer_only or answer_with_thinking.")
        if parts[1] not in {"last_round", "full_history"}:
            raise ValueError("Invalid context_mode. Use last_round or full_history.")
        combinations = [{"output_mode": parts[0], "context_mode": parts[1]}]

    chains: list[dict[str, Any]] = []
    for sample in samples:
        domain = sample["domain"]
        for profile in model_profiles:
            model_key = profile["profile_name"]
            model_name = profile[domain]
            for combo in combinations:
                chain = {
                    "chain_id": make_chain_id(sample["sample_id"], combo, model_key),
                    "sample_id": sample["sample_id"],
                    "dataset_name": sample["dataset_name"],
                    "domain": sample["domain"],
                    "task_type": sample["task_type"],
                    "question": sample["question"],
                    "reference_answer": sample.get("reference_answer", ""),
                    "metadata": sample.get("metadata", {}),
                    "pipeline": combo,
                    "model_profile": model_key,
                    "model_name": model_name,
                    "status": "active",
                    "marked_out_round": None,
                    "consecutive_correct": 0,
                    "nodes": [],
                }
                chains.append(chain)
    chains.sort(key=lambda x: x["chain_id"])
    return chains


def _carry_forward_node(chain: dict[str, Any], round_index: int) -> dict[str, Any]:
    previous = chain["nodes"][-1] if chain["nodes"] else {}
    prev_changed_accum = int(previous.get("changed_correct_to_wrong_accum", 0) or 0)
    return {
        "round": round_index,
        "timestamp": utc_now(),
        "skipped": True,
        "skip_reason": "marked_out",
        "prompt": "",
        "system_prompt": "",
        "prompt_meta": chain["pipeline"],
        "raw_response": "",
        "thinking": previous.get("thinking", ""),
        "answer": previous.get("answer", ""),
        "correct": True,
        "consecutive_correct": chain["consecutive_correct"],
        "marked_out": True,
        "answer_similarity_prev": 1.0,
        "thinking_similarity_prev": 1.0 if previous.get("thinking") else None,
        "changed_correct_to_wrong": 0,
        "changed_correct_to_wrong_accum": prev_changed_accum,
        "progress_score": previous.get("progress_score"),
        "progress_metric": previous.get("progress_metric"),
        "numerical_difference": previous.get("numerical_difference"),
        "cot_monitor_score": previous.get("cot_monitor_score"),
        "trace_score": previous.get("trace_score"),
        "eval_details": {"carried": True},
        "error": "",
    }


def _apply_node(chain: dict[str, Any], node: dict[str, Any], consecutive_k: int) -> None:
    previous = chain["nodes"][-1] if chain["nodes"] else None
    changed_this_round = bool(previous and previous.get("correct") and not node.get("correct"))
    prev_changed_accum = int(previous.get("changed_correct_to_wrong_accum", 0) or 0) if previous else 0
    node["changed_correct_to_wrong"] = int(changed_this_round)
    node["changed_correct_to_wrong_accum"] = prev_changed_accum + int(changed_this_round)

    if node["correct"]:
        chain["consecutive_correct"] += 1
    else:
        chain["consecutive_correct"] = 0

    if chain["status"] == "active" and chain["consecutive_correct"] >= consecutive_k:
        chain["status"] = "marked_out"
        chain["marked_out_round"] = node["round"]

    node["consecutive_correct"] = chain["consecutive_correct"]
    node["marked_out"] = chain["status"] == "marked_out"
    chain["nodes"].append(node)


def compute_round_metrics(chains: list[dict[str, Any]], round_index: int) -> dict[str, Any]:
    node_pairs = [
        (c, c["nodes"][-1])
        for c in chains
        if c["nodes"] and c["nodes"][-1]["round"] == round_index
    ]
    nodes = [node for _chain, node in node_pairs]
    if not node_pairs:
        return {"round": round_index, "num_chains": 0}

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def mean_optional(values: list[float]) -> float | None:
        return mean(values) if values else None

    answer_sims = [x["answer_similarity_prev"] for x in nodes if x["answer_similarity_prev"] is not None]
    think_sims = [x["thinking_similarity_prev"] for x in nodes if x["thinking_similarity_prev"] is not None]
    progress_scores = [float(x["progress_score"]) for x in nodes if x.get("progress_score") is not None]
    math_progress = [
        float(node["progress_score"])
        for chain, node in node_pairs
        if chain.get("task_type") == "math" and node.get("progress_score") is not None
    ]
    numerical_differences = [
        float(node["numerical_difference"])
        for chain, node in node_pairs
        if chain.get("task_type") == "math" and node.get("numerical_difference") is not None
    ]
    code_progress = [
        float(node["progress_score"])
        for chain, node in node_pairs
        if chain.get("task_type") in {"code", "swebench"} and node.get("progress_score") is not None
    ]
    semantic_progress = [
        float(node["progress_score"])
        for chain, node in node_pairs
        if chain.get("task_type") == "text" and node.get("progress_score") is not None
    ]
    cot_scores = [float(x["cot_monitor_score"]) for x in nodes if x.get("cot_monitor_score") is not None]
    trace_scores = [float(x["trace_score"]) for x in nodes if x.get("trace_score") is not None]
    changed_correct_to_wrong_this_round = sum(int(x.get("changed_correct_to_wrong", 0) or 0) for x in nodes)
    changed_correct_to_wrong_accum = sum(int(x.get("changed_correct_to_wrong_accum", 0) or 0) for x in nodes)
    swebench_scores = [
        1.0 if c["nodes"][-1]["correct"] else 0.0
        for c in chains
        if c["nodes"] and c["nodes"][-1]["round"] == round_index and is_swebench_dataset(c["dataset_name"])
    ]

    marked = sum(1 for c in chains if c["status"] == "marked_out")
    active = len(chains) - marked
    return {
        "round": round_index,
        "num_chains": len(chains),
        "round_accuracy": mean([1.0 if x["correct"] else 0.0 for x in nodes]),
        "mean_progress_score": mean_optional(progress_scores),
        "mean_math_progress_score": mean_optional(math_progress),
        "mean_numerical_difference": mean_optional(numerical_differences),
        "mean_code_progress_score": mean_optional(code_progress),
        "mean_semantic_progress_score": mean_optional(semantic_progress),
        "mean_answer_similarity_prev": mean(answer_sims),
        "mean_thinking_similarity_prev": mean(think_sims),
        "mean_cot_monitor_score": mean_optional(cot_scores),
        "mean_trace_score": mean_optional(trace_scores),
        "swebench_resolution_rate": mean_optional(swebench_scores),
        "changed_correct_to_wrong_this_round": int(changed_correct_to_wrong_this_round),
        "changed_correct_to_wrong_accum": int(changed_correct_to_wrong_accum),
        "active_chains": active,
        "marked_out_chains": marked,
    }


def create_experiment_dir(base_dir: Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = hashlib.md5(f"{label}:{stamp}".encode("utf-8")).hexdigest()[:6]
    path = base_dir / f"{stamp}_{label}_{digest}"
    ensure_dir(path)
    ensure_dir(path / "chain")
    ensure_dir(path / "chains")
    ensure_dir(path / "rounds")
    ensure_dir(path / "snapshots")
    ensure_dir(path / "summary")
    return path


def save_chain_states(chains: list[dict[str, Any]], chain_dir: Path) -> None:
    ensure_dir(chain_dir)
    for chain in chains:
        write_json(chain_dir / f"{chain['chain_id']}.json", chain)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: list[float]) -> float | None:
    return _mean(values) if values else None


def _condition_round_metrics_rows(chains: list[dict[str, Any]], round_index: int) -> list[dict[str, Any]]:
    grouped_nodes: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    grouped_status: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"active_chains": 0, "marked_out_chains": 0}
    )

    for chain in chains:
        key = (
            chain["dataset_name"],
            chain["model_name"],
            condition_name(chain["pipeline"]),
        )
        if chain["status"] == "marked_out":
            grouped_status[key]["marked_out_chains"] += 1
        else:
            grouped_status[key]["active_chains"] += 1

        if not chain["nodes"]:
            continue
        node = chain["nodes"][-1]
        if node["round"] != round_index:
            continue
        grouped_nodes[key].append((chain, node))

    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped_nodes.items(), key=lambda x: x[0]):
        dataset_name, model_name, cond = key
        output_mode, context_mode = cond.split("__", 1)
        task_type = str(items[0][0].get("task_type", "")) if items else ""
        nodes = [node for _chain, node in items]
        answer_sims = [n["answer_similarity_prev"] for n in nodes if n["answer_similarity_prev"] is not None]
        think_sims = [n["thinking_similarity_prev"] for n in nodes if n["thinking_similarity_prev"] is not None]
        progress = [float(n["progress_score"]) for n in nodes if n.get("progress_score") is not None]
        math_progress = progress if task_type == "math" else []
        numerical_differences = [float(n["numerical_difference"]) for n in nodes if n.get("numerical_difference") is not None]
        code_progress = progress if task_type in {"code", "swebench"} else []
        semantic_progress = progress if task_type == "text" else []
        cot_scores = [float(n["cot_monitor_score"]) for n in nodes if n.get("cot_monitor_score") is not None]
        trace_scores = [float(n["trace_score"]) for n in nodes if n.get("trace_score") is not None]
        changed_correct_to_wrong_this_round = int(
            sum(int(n.get("changed_correct_to_wrong", 0) or 0) for n in nodes)
        )
        changed_correct_to_wrong_accum = int(
            sum(int(n.get("changed_correct_to_wrong_accum", 0) or 0) for n in nodes)
        )
        resolution_rate = (
            _mean([1.0 if n["correct"] else 0.0 for n in nodes]) if is_swebench_dataset(dataset_name) else None
        )

        row = {
            "round": round_index,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "condition": cond,
            "output_mode": output_mode,
            "context_mode": context_mode,
            "num_questions": len(nodes),
            "round_accuracy": _mean([1.0 if n["correct"] else 0.0 for n in nodes]),
            "mean_progress_score": _mean_optional(progress),
            "mean_math_progress_score": _mean_optional(math_progress),
            "mean_numerical_difference": _mean_optional(numerical_differences),
            "mean_code_progress_score": _mean_optional(code_progress),
            "mean_semantic_progress_score": _mean_optional(semantic_progress),
            "mean_answer_similarity_prev": _mean(answer_sims),
            "mean_thinking_similarity_prev": _mean(think_sims),
            "mean_cot_monitor_score": _mean_optional(cot_scores),
            "mean_trace_score": _mean_optional(trace_scores),
            "swebench_resolution_rate": resolution_rate,
            "changed_correct_to_wrong_this_round": changed_correct_to_wrong_this_round,
            "changed_correct_to_wrong_accum": changed_correct_to_wrong_accum,
            "active_chains": grouped_status[key]["active_chains"],
            "marked_out_chains": grouped_status[key]["marked_out_chains"],
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_summary_dataframes(chains: list[dict[str, Any]], exp_dir: Path, max_rounds: int) -> None:
    summary_dir = exp_dir / "summary"
    ensure_dir(summary_dir)

    detailed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for chain in chains:
        cond = condition_name(chain["pipeline"])
        output_mode, context_mode = cond.split("__", 1)
        qid = sample_question_id(chain["sample_id"])
        for node in chain["nodes"]:
            resolution_rate = float(node["correct"]) if is_swebench_dataset(chain["dataset_name"]) else None
            row = {
                "max_rounds": max_rounds,
                "dataset_name": chain["dataset_name"],
                "model_name": chain["model_name"],
                "model_profile": chain["model_profile"],
                "condition": cond,
                "output_mode": output_mode,
                "context_mode": context_mode,
                "sample_id": chain["sample_id"],
                "question_id": qid,
                "round": node["round"],
                "correct": int(node["correct"]),
                "progress_score": node.get("progress_score"),
                "progress_metric": node.get("progress_metric"),
                "numerical_difference": node.get("numerical_difference"),
                "answer_similarity_prev": node["answer_similarity_prev"],
                "thinking_similarity_prev": node["thinking_similarity_prev"],
                "cot_monitor_score": node["cot_monitor_score"],
                "trace_score": node["trace_score"],
                "resolution_rate": resolution_rate,
                "changed_correct_to_wrong": int(node.get("changed_correct_to_wrong", 0)),
                "changed_correct_to_wrong_accum": int(node.get("changed_correct_to_wrong_accum", 0)),
                "marked_out": int(node.get("marked_out", False)),
                "consecutive_correct": node.get("consecutive_correct", 0),
            }
            detailed_rows.append(row)
            grouped[(chain["dataset_name"], chain["model_name"], cond, node["round"])].append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda x: x[0]):
        dataset_name, model_name, cond, round_index = key
        output_mode, context_mode = cond.split("__", 1)
        aggregated_rows.append(
            {
                "max_rounds": max_rounds,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "condition": cond,
                "output_mode": output_mode,
                "context_mode": context_mode,
                "round": round_index,
                "num_questions": len(rows),
                "round_accuracy": _mean([float(r["correct"]) for r in rows]),
                "mean_progress_score": _mean_optional(
                    [float(r["progress_score"]) for r in rows if r.get("progress_score") is not None]
                ),
                "mean_math_progress_score": (
                    _mean_optional([float(r["progress_score"]) for r in rows if r.get("progress_score") is not None])
                    if any("math" in str(r.get("progress_metric", "")) for r in rows)
                    else None
                ),
                "mean_numerical_difference": _mean_optional(
                    [float(r["numerical_difference"]) for r in rows if r.get("numerical_difference") is not None]
                ),
                "mean_code_progress_score": (
                    _mean_optional([float(r["progress_score"]) for r in rows if r.get("progress_score") is not None])
                    if any(
                        str(r.get("progress_metric", "")).startswith("code")
                        or str(r.get("progress_metric", "")).startswith("swebench")
                        for r in rows
                    )
                    else None
                ),
                "mean_semantic_progress_score": (
                    _mean_optional([float(r["progress_score"]) for r in rows if r.get("progress_score") is not None])
                    if any(str(r.get("progress_metric", "")).startswith("semantic") for r in rows)
                    else None
                ),
                "mean_answer_similarity_prev": _mean(
                    [float(r["answer_similarity_prev"]) for r in rows if r["answer_similarity_prev"] is not None]
                ),
                "mean_thinking_similarity_prev": _mean(
                    [float(r["thinking_similarity_prev"]) for r in rows if r["thinking_similarity_prev"] is not None]
                ),
                "mean_cot_monitor_score": _mean_optional(
                    [float(r["cot_monitor_score"]) for r in rows if r["cot_monitor_score"] is not None]
                ),
                "mean_trace_score": _mean_optional(
                    [float(r["trace_score"]) for r in rows if r["trace_score"] is not None]
                ),
                "swebench_resolution_rate": _mean_optional(
                    [float(r["resolution_rate"]) for r in rows if r["resolution_rate"] is not None]
                ),
                "changed_correct_to_wrong_this_round": int(
                    sum(int(r["changed_correct_to_wrong"]) for r in rows)
                ),
                "changed_correct_to_wrong_accum": int(
                    sum(int(r["changed_correct_to_wrong_accum"]) for r in rows)
                ),
                "marked_out_questions": int(sum(int(r["marked_out"]) for r in rows)),
            }
        )

    summary_jsonl = summary_dir / "summary_dataframe.jsonl"
    if summary_jsonl.exists():
        summary_jsonl.unlink()
    for row in detailed_rows:
        append_jsonl(summary_jsonl, row)

    aggregated_jsonl = summary_dir / "summary_round_by_condition.jsonl"
    if aggregated_jsonl.exists():
        aggregated_jsonl.unlink()
    for row in aggregated_rows:
        append_jsonl(aggregated_jsonl, row)

    _write_csv(summary_dir / "summary_dataframe.csv", detailed_rows)
    _write_csv(summary_dir / "summary_round_by_condition.csv", aggregated_rows)


def save_round_outputs(
    exp_dir: Path,
    chains: list[dict[str, Any]],
    round_index: int,
    round_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    round_file = exp_dir / "rounds" / f"round_{round_index:03d}.jsonl"
    condition_metrics = _condition_round_metrics_rows(chains, round_index)

    for chain in chains:
        node = chain["nodes"][-1]
        payload = {
            "chain_id": chain["chain_id"],
            "sample_id": chain["sample_id"],
            "dataset_name": chain["dataset_name"],
            "domain": chain["domain"],
            "task_type": chain["task_type"],
            "model_profile": chain["model_profile"],
            "model_name": chain["model_name"],
            "pipeline": chain["pipeline"],
            "node": node,
        }
        append_jsonl(round_file, payload)
        record_dir = condition_dir(exp_dir, chain)
        append_jsonl(record_dir / data_file_name(chain["sample_id"]), payload)

    for row in condition_metrics:
        metrics_payload = {
            "round": row["round"],
            "dataset_name": row["dataset_name"],
            "model_name": row["model_name"],
            "condition": row["condition"],
            "output_mode": row["output_mode"],
            "context_mode": row["context_mode"],
            "num_questions": row["num_questions"],
            "round_accuracy": row["round_accuracy"],
            "mean_progress_score": row["mean_progress_score"],
            "mean_math_progress_score": row["mean_math_progress_score"],
            "mean_numerical_difference": row["mean_numerical_difference"],
            "mean_code_progress_score": row["mean_code_progress_score"],
            "mean_semantic_progress_score": row["mean_semantic_progress_score"],
            "mean_answer_similarity_prev": row["mean_answer_similarity_prev"],
            "mean_thinking_similarity_prev": row["mean_thinking_similarity_prev"],
            "mean_cot_monitor_score": row["mean_cot_monitor_score"],
            "mean_trace_score": row["mean_trace_score"],
            "swebench_resolution_rate": row["swebench_resolution_rate"],
            "changed_correct_to_wrong_this_round": row["changed_correct_to_wrong_this_round"],
            "changed_correct_to_wrong_accum": row["changed_correct_to_wrong_accum"],
            "active_chains": row["active_chains"],
            "marked_out_chains": row["marked_out_chains"],
        }
        condition_parts = {
            "dataset_name": row["dataset_name"],
            "model_name": row["model_name"],
            "pipeline": {
                "output_mode": row["output_mode"],
                "context_mode": row["context_mode"],
            },
        }
        append_jsonl(condition_dir(exp_dir, condition_parts) / "metrics_round.jsonl", metrics_payload)
        append_jsonl(exp_dir / "rounds" / "condition_metrics_round.jsonl", metrics_payload)

    append_jsonl(exp_dir / "rounds" / "round_metrics.jsonl", round_metrics)

    snapshot = {
        "round": round_index,
        "round_metrics": round_metrics,
        "chains": chains,
    }
    write_json(exp_dir / "snapshots" / f"round_{round_index:03d}.json", snapshot)
    return condition_metrics


def _latest_round_from_snapshots(exp_dir: Path) -> int:
    snaps = sorted((exp_dir / "snapshots").glob("round_*.json"))
    if not snaps:
        return 0
    last = snaps[-1].stem
    return int(last.split("_")[-1])


def load_resume_snapshot(exp_dir: Path, resume_round: int | None) -> tuple[list[dict[str, Any]], int]:
    if resume_round is None:
        resume_round = _latest_round_from_snapshots(exp_dir)
    if resume_round <= 0:
        raise ValueError("No snapshot available for resume.")
    path = exp_dir / "snapshots" / f"round_{resume_round:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    payload = read_json(path)
    return payload["chains"], int(payload["round"])


def run_chain_experiment(
    *,
    chains: list[dict[str, Any]],
    exp_dir: Path,
    max_rounds: int,
    consecutive_k: int,
    test_timeout: int,
    logger,
    inference_fn: Callable[[dict[str, Any], str, str, int], tuple[str, str]],
    workers: int = 1,
    start_round: int = 1,
    continue_marked_chains: bool = False,
    compute_reasoning_metrics: bool = True,
    cot_monitor_fn: Callable[[dict[str, Any], str, str, bool, int], float] | None = None,
    trace_score_fn: Callable[[dict[str, Any], str, str, bool, int, int], float] | None = None,
    round_hook: Callable[[int, dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> None:
    save_chain_states(chains, exp_dir / "chains")

    for round_index in range(start_round, max_rounds + 1):
        logger.info("Starting round %s with %s chains", round_index, len(chains))
        active_payloads: dict[str, tuple[str, str, dict[str, Any]]] = {}

        def _should_infer(chain_obj: dict[str, Any]) -> bool:
            if chain_obj["status"] == "active":
                return True
            if continue_marked_chains and chain_obj["status"] == "marked_out":
                return True
            return False

        for chain in chains:
            if not _should_infer(chain):
                continue
            prompt, system_prompt, prompt_meta = build_prompt(chain, round_index)
            active_payloads[chain["chain_id"]] = (prompt, system_prompt, prompt_meta)

        responses: dict[str, tuple[str, str]] = {}

        def _call(chain_obj: dict[str, Any], prompt_text: str, sys_prompt: str):
            return inference_fn(chain_obj, prompt_text, sys_prompt, round_index)

        active_chains = [c for c in chains if _should_infer(c)]
        if workers > 1 and len(active_chains) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {}
                for chain in active_chains:
                    prompt_text, sys_prompt, _meta = active_payloads[chain["chain_id"]]
                    fut = executor.submit(_call, chain, prompt_text, sys_prompt)
                    future_map[fut] = chain["chain_id"]
                for future in as_completed(future_map):
                    chain_id = future_map[future]
                    try:
                        responses[chain_id] = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        responses[chain_id] = ("", str(exc))
        else:
            for chain in active_chains:
                prompt_text, sys_prompt, _meta = active_payloads[chain["chain_id"]]
                try:
                    responses[chain["chain_id"]] = _call(chain, prompt_text, sys_prompt)
                except Exception as exc:  # pragma: no cover - defensive
                    responses[chain["chain_id"]] = ("", str(exc))

        for chain in chains:
            if not _should_infer(chain):
                node = _carry_forward_node(chain, round_index)
                chain["nodes"].append(node)
                continue

            prompt, system_prompt, prompt_meta = active_payloads[chain["chain_id"]]
            raw_response, error_text = responses.get(chain["chain_id"], ("", "missing response"))
            thinking, answer = parse_model_output(
                raw_response,
                chain["pipeline"]["output_mode"],
                chain["task_type"],
            )

            correct, eval_details = evaluate_answer(
                chain,
                answer,
                test_timeout=test_timeout,
                thinking=thinking,
            )
            prev = chain["nodes"][-1] if chain["nodes"] else None
            answer_sim = (
                text_similarity(prev.get("answer", ""), answer) if prev is not None else None
            )
            if prev is not None and (thinking or prev.get("thinking", "")):
                think_sim = text_similarity(prev.get("thinking", ""), thinking)
            else:
                think_sim = None

            if compute_reasoning_metrics:
                if cot_monitor_fn is not None:
                    try:
                        cot_score = clamp(float(cot_monitor_fn(chain, answer, thinking, bool(correct), round_index)))
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        logger.warning("COT monitor callback failed for chain %s: %s", chain["chain_id"], exc)
                        cot_score = compute_cot_monitor_score(chain["question"], answer, thinking, correct)
                else:
                    cot_score = compute_cot_monitor_score(chain["question"], answer, thinking, correct)

                if trace_score_fn is not None:
                    try:
                        trace_score = clamp(
                            float(trace_score_fn(chain, answer, thinking, bool(correct), round_index, test_timeout))
                        )
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        logger.warning("TRACE callback failed for chain %s: %s", chain["chain_id"], exc)
                        trace_score = compute_trace_score(answer, thinking, correct)
                else:
                    trace_score = compute_trace_score(answer, thinking, correct)
            else:
                cot_score = None
                trace_score = None

            node = {
                "round": round_index,
                "timestamp": utc_now(),
                "skipped": False,
                "skip_reason": "",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "prompt_meta": prompt_meta,
                "raw_response": raw_response,
                "thinking": thinking,
                "answer": answer,
                "correct": bool(correct),
                "answer_similarity_prev": answer_sim,
                "thinking_similarity_prev": think_sim,
                "progress_score": eval_details.get("progress_score"),
                "progress_metric": eval_details.get("progress_metric"),
                "numerical_difference": eval_details.get("numerical_difference"),
                "cot_monitor_score": cot_score,
                "trace_score": trace_score,
                "eval_details": eval_details,
                "error": error_text,
            }
            _apply_node(chain, node, consecutive_k=consecutive_k)

        round_metrics = compute_round_metrics(chains, round_index)
        save_chain_states(chains, exp_dir / "chains")
        condition_metrics = save_round_outputs(exp_dir, chains, round_index, round_metrics)
        if round_hook is not None:
            try:
                round_hook(round_index, round_metrics, condition_metrics)
            except Exception as exc:  # pragma: no cover - monitoring should not break run
                logger.warning("Round hook failed at round %s: %s", round_index, exc)
        write_json(
            exp_dir / "state.json",
            {
                "last_completed_round": round_index,
                "max_rounds": max_rounds,
                "consecutive_k": consecutive_k,
                "continue_marked_chains": continue_marked_chains,
                "compute_reasoning_metrics": compute_reasoning_metrics,
                "total_chains": len(chains),
                "timestamp": utc_now(),
            },
        )
        logger.info(
            "Round %s complete: accuracy=%.4f active=%s marked=%s",
            round_index,
            round_metrics.get("round_accuracy", 0.0),
            round_metrics.get("active_chains", 0),
            round_metrics.get("marked_out_chains", 0),
        )

    _save_summary_dataframes(chains, exp_dir, max_rounds)


def save_dataset_snapshot(samples: list[dict[str, Any]], exp_dir: Path) -> None:
    path = exp_dir / "dataset_snapshot.jsonl"
    for sample in samples:
        append_jsonl(path, sample)


def load_benchmark_samples(
    manifest_path: Path,
    selected_datasets: list[str] | None,
    max_samples_per_dataset: int,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    datasets = manifest.get("datasets", [])
    selected_exact: set[str] | None = None
    selected_canonical: set[str] | None = None
    if selected_datasets:
        selected_exact = set()
        selected_canonical = set()
        for raw in selected_datasets:
            name = raw.strip().lower()
            if not name:
                continue
            selected_exact.add(name)
            selected_canonical.add(canonical_dataset_name(name))

    rows: list[dict[str, Any]] = []
    matched_canonical: set[str] = set()
    for dataset in datasets:
        name = str(dataset["dataset_name"])
        name_l = name.lower()
        name_canonical = canonical_dataset_name(name_l)
        if selected_exact is not None and selected_canonical is not None:
            if name_l not in selected_exact and name_canonical not in selected_canonical:
                continue
        matched_canonical.add(name_canonical)
        file_path = manifest_path.parent / dataset["file"]
        data = read_jsonl(file_path)
        for item in data[:max_samples_per_dataset]:
            rows.append(item)

    if selected_canonical is not None:
        missing = sorted(selected_canonical - matched_canonical)
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Selected dataset(s) not found in manifest {manifest_path}: {missing_text}. "
                "Regenerate the benchmark manifest or adjust --datasets."
            )
    return rows
