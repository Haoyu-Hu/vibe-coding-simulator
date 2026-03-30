"""Vibe-coding tester role.

The tester is an LLM-backed evaluator that:
1. Reads the original task and current code artifact.
2. Infers what the user likely cares about.
3. Generates a small set of PUBLIC tests (shown to user) and a larger set of HIDDEN tests.
4. Executes all tests via subprocess.
5. Returns a restricted public feedback summary to the user.
6. Records all hidden test results as experiment metrics.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from utils.eval_code import extract_python_code

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    description: str
    code: str       # Python executable assertion, e.g. "assert f(1) == 2"
    is_public: bool = True


@dataclass
class TestResult:
    description: str
    code: str
    passed: bool
    stdout: str
    stderr: str
    error: str = ""


@dataclass
class TesterOutput:
    """All outputs from one tester invocation."""
    # Public tests (shown to user, limited count)
    public_tests: list[TestCase] = field(default_factory=list)
    public_results: list[TestResult] = field(default_factory=list)
    public_pass_rate: float = 0.0
    public_test_count: int = 0
    # Hidden tests (private, broader coverage)
    hidden_tests: list[TestCase] = field(default_factory=list)
    hidden_results: list[TestResult] = field(default_factory=list)
    hidden_pass_rate: float = 0.0
    hidden_test_count: int = 0
    # Public feedback message (what the user simulator receives)
    public_feedback: str = ""
    # Tester LLM raw output (for logging)
    tester_raw_output: str = ""
    tester_prompt: str = ""
    tester_error: str = ""
    # Failure categories
    has_syntax_error: bool = False
    has_runtime_error: bool = False
    # Regression tracking (set by caller)
    hidden_regression_count: int = 0


@dataclass
class VibeCodingTesterConfig:
    """Configuration for the tester role."""
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1200
    request_timeout: int = 90
    public_test_budget: int = 4
    hidden_test_budget: int = 12
    test_timeout: int = 15
    max_feedback_words: int = 80


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_TESTER_SYSTEM_PROMPT = (
    "You are a software tester. Your job is to evaluate whether a piece of Python code "
    "correctly solves the user's task. You must generate TWO sets of tests:\n"
    "1. PUBLIC TESTS: a small set (2-4) of representative test cases that a non-expert user "
    "could understand and that reveal the most important issues.\n"
    "2. HIDDEN TESTS: a broader set (5-15) of test cases including edge cases, boundary values, "
    "and failure modes. These are for private evaluation only.\n"
    "Also write a short PUBLIC FEEDBACK message (max 3 sentences) that a real user would pass "
    "back to the developer — limited, human-usable, no dump of all test cases.\n"
    "Respond ONLY with valid JSON in the exact schema below. No prose outside the JSON.\n"
    '{\n'
    '  "reasoning": "...",\n'
    '  "public_tests": [\n'
    '    {"description": "...", "code": "assert ..."},\n'
    '    ...\n'
    '  ],\n'
    '  "hidden_tests": [\n'
    '    {"description": "...", "code": "assert ..."},\n'
    '    ...\n'
    '  ],\n'
    '  "public_feedback": "..."\n'
    '}'
)


def build_tester_prompt(
    *,
    chain: dict[str, Any],
    code_artifact: str,
    round_index: int,
    user_messages: list[str],
    public_test_budget: int,
    hidden_test_budget: int,
) -> str:
    original_task = str(chain.get("question", "")).strip()
    metadata = chain.get("metadata", {})
    entry_point = str(metadata.get("entry_point", "")).strip()
    reference_tests = str(metadata.get("test", "")).strip()

    lines = [
        f"ORIGINAL TASK (round 1 user intent):",
        original_task,
        "",
    ]

    if user_messages:
        lines += [
            "USER MESSAGES SO FAR (in order):",
        ]
        for i, msg in enumerate(user_messages[-3:], start=1):
            lines.append(f"  User message {i}: {str(msg).strip()[:300]}")
        lines.append("")

    if entry_point:
        lines += [f"ENTRY POINT: {entry_point}", ""]

    lines += [
        f"CODE TO EVALUATE (round {round_index}):",
        "```python",
        (code_artifact.strip() if code_artifact.strip() else "# (no code extracted)"),
        "```",
        "",
        f"Generate {public_test_budget} public tests and up to {hidden_test_budget} hidden tests.",
        "Each test 'code' field must be a standalone Python expression (e.g., assert f(1) == 2).",
        "Public feedback must be short (2-3 sentences max) and human-readable.",
    ]

    if reference_tests:
        lines += [
            "",
            "REFERENCE TEST HINTS (use these to guide hidden tests, do not copy verbatim):",
            reference_tests[:500],
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    # Try to find a JSON block in the response
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    # Try to find a raw JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


def _parse_tester_output(raw_text: str) -> dict[str, Any] | None:
    raw = _extract_json_block(raw_text)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _coerce_test_list(items: Any) -> list[TestCase]:
    cases: list[TestCase] = []
    if not isinstance(items, list):
        return cases
    for item in items:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        code = str(item.get("code", "")).strip()
        if code:
            cases.append(TestCase(description=desc, code=code))
    return cases


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def _run_single_test(
    candidate_code: str,
    test_code: str,
    timeout: int,
) -> tuple[bool, str, str]:
    """Execute a single test assertion against the candidate code.

    Returns: (passed, stdout, stderr)
    """
    full_code = candidate_code.rstrip() + "\n\n" + test_code.strip() + "\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_vibe.py")
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(full_code)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("MPLCONFIGDIR", tmpdir)
        try:
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {timeout}s"
    return result.returncode == 0, result.stdout[:800], result.stderr[:800]


def _run_test_cases(
    candidate_code: str,
    test_cases: list[TestCase],
    timeout: int,
) -> list[TestResult]:
    results: list[TestResult] = []
    for tc in test_cases:
        passed, stdout, stderr = _run_single_test(candidate_code, tc.code, timeout)
        has_syntax = "SyntaxError" in stderr or "IndentationError" in stderr
        results.append(
            TestResult(
                description=tc.description,
                code=tc.code,
                passed=passed,
                stdout=stdout,
                stderr=stderr,
                error="syntax_error" if has_syntax else ("runtime_error" if stderr.strip() else ""),
            )
        )
    return results


def _pass_rate(results: list[TestResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


# ---------------------------------------------------------------------------
# Public feedback generation (fallback when tester LLM fails to produce it)
# ---------------------------------------------------------------------------

def _fallback_public_feedback(
    public_results: list[TestResult],
    hidden_results: list[TestResult],
) -> str:
    all_results = public_results or hidden_results
    if not all_results:
        return "The code could not be evaluated due to a tester error."
    failing = [r for r in all_results if not r.passed]
    passing = [r for r in all_results if r.passed]
    if not failing:
        return "All tested cases passed. The implementation looks correct."
    first_fail = failing[0]
    stderr_hint = first_fail.stderr.strip().splitlines()[-1][:80] if first_fail.stderr.strip() else ""
    msg = f"Found {len(failing)} failing case(s)."
    if first_fail.description:
        msg += f" For example: {first_fail.description}."
    if stderr_hint:
        msg += f" Error: {stderr_hint}"
    return msg


# ---------------------------------------------------------------------------
# Main tester entry point
# ---------------------------------------------------------------------------

TesterInferenceFn = Callable[[str, str], tuple[str, str]]


def run_tester(
    *,
    chain: dict[str, Any],
    code_artifact: str,
    round_index: int,
    user_messages: list[str],
    tester_inference_fn: TesterInferenceFn | None,
    tester_config: VibeCodingTesterConfig,
    logger,
) -> TesterOutput:
    """Run the tester role for one round.

    Returns a TesterOutput with public and hidden test results.
    """
    out = TesterOutput()

    prompt = build_tester_prompt(
        chain=chain,
        code_artifact=code_artifact,
        round_index=round_index,
        user_messages=user_messages,
        public_test_budget=tester_config.public_test_budget,
        hidden_test_budget=tester_config.hidden_test_budget,
    )
    out.tester_prompt = prompt

    raw_text = ""
    tester_error = ""

    if tester_inference_fn is not None:
        try:
            raw_text, tester_error = tester_inference_fn(prompt, _TESTER_SYSTEM_PROMPT)
        except Exception as exc:
            tester_error = f"{exc.__class__.__name__}: {exc}"

    if tester_error and logger is not None:
        logger.warning(
            "Tester error on chain %s round %s: %s",
            chain.get("chain_id", "?"),
            round_index,
            tester_error,
        )

    out.tester_raw_output = raw_text
    out.tester_error = tester_error

    # Parse tester JSON output
    parsed = _parse_tester_output(raw_text) if raw_text.strip() else None

    if parsed is not None:
        public_cases = _coerce_test_list(parsed.get("public_tests", []))[: tester_config.public_test_budget]
        hidden_cases = _coerce_test_list(parsed.get("hidden_tests", []))[: tester_config.hidden_test_budget]
        public_feedback_raw = str(parsed.get("public_feedback", "")).strip()
    else:
        public_cases = []
        hidden_cases = []
        public_feedback_raw = ""

    # If no tests were generated, create minimal fallback tests from reference
    if not public_cases and not hidden_cases:
        public_cases, hidden_cases = _generate_fallback_tests(chain, code_artifact)

    out.public_tests = public_cases
    out.hidden_tests = hidden_cases

    # Execute tests if we have code
    if code_artifact.strip():
        out.public_results = _run_test_cases(code_artifact, public_cases, tester_config.test_timeout)
        out.hidden_results = _run_test_cases(code_artifact, hidden_cases, tester_config.test_timeout)
    else:
        # No code - all tests fail
        out.public_results = [
            TestResult(description=tc.description, code=tc.code, passed=False, stdout="", stderr="no code")
            for tc in public_cases
        ]
        out.hidden_results = [
            TestResult(description=tc.description, code=tc.code, passed=False, stdout="", stderr="no code")
            for tc in hidden_cases
        ]

    out.public_test_count = len(out.public_results)
    out.hidden_test_count = len(out.hidden_results)
    out.public_pass_rate = _pass_rate(out.public_results)
    out.hidden_pass_rate = _pass_rate(out.hidden_results)

    # Check for error categories
    all_results = out.public_results + out.hidden_results
    out.has_syntax_error = any("syntax_error" in r.error for r in all_results)
    out.has_runtime_error = any("runtime_error" in r.error for r in all_results)

    # Public feedback
    if public_feedback_raw:
        out.public_feedback = public_feedback_raw
    else:
        out.public_feedback = _fallback_public_feedback(out.public_results, out.hidden_results)

    return out


# ---------------------------------------------------------------------------
# Fallback test generation (from reference tests in dataset metadata)
# ---------------------------------------------------------------------------

def _extract_assert_lines_from_reference(test_code: str) -> list[str]:
    lines = []
    for raw in test_code.splitlines():
        line = raw.strip()
        if line.startswith("assert "):
            lines.append(line)
    return lines


def _generate_fallback_tests(
    chain: dict[str, Any],
    code_artifact: str,
) -> tuple[list[TestCase], list[TestCase]]:
    """Generate minimal tests from dataset reference test code when the tester LLM fails."""
    metadata = chain.get("metadata", {})
    test_code = str(metadata.get("test", "")).strip()
    if not test_code:
        return [], []

    assert_lines = _extract_assert_lines_from_reference(test_code)
    if not assert_lines:
        return [], []

    # Split: first 2 as public, rest as hidden
    split = min(2, len(assert_lines))
    public_cases = [
        TestCase(description=f"Reference test {i+1}", code=line, is_public=True)
        for i, line in enumerate(assert_lines[:split])
    ]
    hidden_cases = [
        TestCase(description=f"Reference test {i+1+split}", code=line, is_public=False)
        for i, line in enumerate(assert_lines[split:])
    ]
    return public_cases, hidden_cases


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def test_case_to_dict(tc: TestCase) -> dict[str, Any]:
    return {"description": tc.description, "code": tc.code, "is_public": tc.is_public}


def test_result_to_dict(tr: TestResult) -> dict[str, Any]:
    return {
        "description": tr.description,
        "code": tr.code,
        "passed": tr.passed,
        "stdout": tr.stdout[:500],
        "stderr": tr.stderr[:500],
        "error": tr.error,
    }


def tester_output_to_dict(out: TesterOutput) -> dict[str, Any]:
    return {
        "public_tests": [test_case_to_dict(tc) for tc in out.public_tests],
        "public_results": [test_result_to_dict(tr) for tr in out.public_results],
        "public_pass_rate": out.public_pass_rate,
        "public_test_count": out.public_test_count,
        "hidden_tests": [test_case_to_dict(tc) for tc in out.hidden_tests],
        "hidden_results": [test_result_to_dict(tr) for tr in out.hidden_results],
        "hidden_pass_rate": out.hidden_pass_rate,
        "hidden_test_count": out.hidden_test_count,
        "public_feedback": out.public_feedback,
        "tester_raw_output": out.tester_raw_output[:2000],
        "tester_prompt": out.tester_prompt[:2000],
        "tester_error": out.tester_error,
        "has_syntax_error": out.has_syntax_error,
        "has_runtime_error": out.has_runtime_error,
        "hidden_regression_count": out.hidden_regression_count,
    }
