from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass


@dataclass
class CodeEvalResult:
    passed: bool
    stdout: str
    stderr: str


def strip_markdown(code: str) -> str:
    text = code.strip()
    if not text:
        return ""

    fenced_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        return fenced_blocks[0].strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text)
        text = text.strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _is_likely_prose(line: str) -> bool:
    value = line.strip()
    if not value:
        return False
    if value.startswith(("#", "def ", "class ", "@", "from ", "import ", "return ", "if ", "for ", "while ", "try", "except", "with ", "pass", "raise", "assert ", "elif ", "else:", "finally:")):
        return False
    if value.startswith(("'''", '"""')) or value.endswith(("'''", '"""')):
        return False
    if re.match(r"^[\]\)\}:,.\s]+$", value):
        return False
    if re.search(r"[=\(\)\[\]\{\}:]", value):
        return False
    if value.startswith(("-", "*")) and len(value.split()) > 2:
        return True
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9 ,.'`-]{6,}$", value))


def _trim_non_code_tail(text: str) -> str:
    lines = text.splitlines()
    while lines and _is_likely_prose(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _slice_from_code_start(text: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s*(def|class|from|import|@)\b", line):
            start = idx
            break
    if start is None:
        return text
    return "\n".join(lines[start:]).strip()


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    value = text.strip()

    value = re.sub(r"^\s*answer\s*:\s*", "", value, flags=re.IGNORECASE)
    value = strip_markdown(value)
    value = _slice_from_code_start(value)
    value = _trim_non_code_tail(value)

    lines = value.splitlines()
    first_code_line = next((line for line in lines if line.strip()), "")
    # Function-body style completions (common in BigCodeBench/HumanEval references)
    # are not parseable as standalone modules. Keep them intact instead of truncating
    # to a shorter parseable prefix.
    if first_code_line and not re.match(r"^\s*(def|class|from|import|@)\b", first_code_line):
        if any(line.startswith((" ", "\t")) for line in lines[1:]):
            return value.strip()

    # If we can parse a prefix as Python, keep the longest valid prefix.
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            # Model-generated code often contains regex/docstring escapes like \s or \.
            # Ignore parser SyntaxWarning here to avoid noisy logs during artifact extraction.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return value.strip()


def build_candidate(prompt: str, completion: str, entry_point: str | None = None) -> str:
    completion = extract_python_code(completion)
    prompt_scaffold = _slice_from_code_start(strip_markdown(prompt))
    if not prompt_scaffold.strip():
        prompt_scaffold = prompt
    if entry_point:
        pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\("
        if re.search(pattern, completion, flags=re.MULTILINE):
            return completion

    lines = completion.splitlines()
    if lines and lines[0] and not lines[0].startswith((" ", "\t")):
        # Body-only completions often keep residual indentation from the original
        # function context. Normalize that indentation before nesting under prompt.
        tail_indents = []
        for line in lines[1:]:
            if not line.strip():
                continue
            match = re.match(r"^[ \t]+", line)
            if match:
                tail_indents.append(len(match.group(0)))
        if tail_indents:
            trim = min(tail_indents)
            normalized = [lines[0]]
            for line in lines[1:]:
                if line.strip() and len(line) >= trim:
                    normalized.append(line[trim:])
                else:
                    normalized.append(line)
            lines = normalized
        lines = [f"    {line}" if line.strip() else "" for line in lines]
    completion = "\n".join(lines)
    if completion and not completion.startswith("\n"):
        completion = "\n" + completion
    return prompt_scaffold + completion


def run_tests(candidate_code: str, test_code: str, entry_point: str, timeout: int) -> CodeEvalResult:
    full_code = candidate_code + "\n\n" + test_code + "\n\n"
    # HumanEval-style tests define a check(candidate) function.
    # MBPP-style tests are direct assert statements and should not call check().
    if re.search(r"^\s*def\s+check\s*\(", test_code, flags=re.MULTILINE):
        full_code += f"check({entry_point})\n"
    # BigCodeBench-style tests define unittest.TestCase classes without invoking unittest.main().
    # Execute the suite explicitly so assertions are actually evaluated.
    elif re.search(r"\bunittest\.TestCase\b", test_code):
        full_code += (
            "if __name__ == '__main__':\n"
            "    import unittest\n"
            "    unittest.main()\n"
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "candidate.py")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(full_code)
        try:
            env = os.environ.copy()
            # Force non-interactive plotting backend so generated code cannot pop GUI windows.
            env.setdefault("MPLBACKEND", "Agg")
            env.setdefault("MPLCONFIGDIR", tmpdir)
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return CodeEvalResult(False, "", f"Timeout after {timeout}s")
    return CodeEvalResult(result.returncode == 0, result.stdout, result.stderr)
