"""Vibe-coding user simulator.

Simulates a human developer-user at one of six knowledge levels who interacts
with an AI code developer across multiple rounds.  The level is sampled once per
chain and kept fixed; it controls how the user frames requests, inspects code,
and gives feedback.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Level constants
# ---------------------------------------------------------------------------

KNOWLEDGE_LEVELS = [1, 2, 3, 4, 5, 6]

LEVEL_LABELS: dict[int, str] = {
    1: "no coding background",
    2: "low coding literacy",
    3: "beginner coder",
    4: "intermediate coder",
    5: "advanced coder",
    6: "expert developer",
}

# Default weight vector (normal-like, centered near 3-4).
# Stored here as fallback; canonical values are in coder-simulate-rubric/defaults.json.
DEFAULT_LEVEL_WEIGHTS = [0.05, 0.15, 0.25, 0.30, 0.15, 0.10]

_RUBRIC_DIR = Path(__file__).parent.parent / "coder-simulate-rubric"

_RUBRIC_CACHE: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Rubric loading
# ---------------------------------------------------------------------------

def _load_rubric_text(level: int) -> str:
    if level in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[level]
    path = _RUBRIC_DIR / f"level_{level}.md"
    text = ""
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
    _RUBRIC_CACHE[level] = text
    return text


def load_rubric_defaults() -> dict[str, Any]:
    path = _RUBRIC_DIR / "defaults.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Level sampling
# ---------------------------------------------------------------------------

def _stable_hash(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def sample_knowledge_level(chain_id: str, weights: list[float] | None = None) -> int:
    """Sample a knowledge level deterministically from chain_id.

    The level is derived entirely from the chain_id so that resuming a run
    produces the same level for every chain.
    """
    if weights is None:
        weights = DEFAULT_LEVEL_WEIGHTS
    total = sum(weights)
    norm = [w / total for w in weights]
    seed_val = _stable_hash(f"vibe-coding-level:{chain_id}")
    r = (seed_val % 1_000_000) / 1_000_000.0
    cumulative = 0.0
    for idx, w in enumerate(norm):
        cumulative += w
        if r < cumulative:
            return KNOWLEDGE_LEVELS[idx]
    return KNOWLEDGE_LEVELS[-1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VibeCodingUserConfig:
    """Configuration for the vibe-coding user simulator."""
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 300
    request_timeout: int = 60
    word_limit: int = 120
    memory_window: int = 2
    # Weight vector for levels 1-6; None means use DEFAULT_LEVEL_WEIGHTS
    level_weights: list[float] | None = None


# ---------------------------------------------------------------------------
# Compact rubric summaries (inline fallback when rubric files are unavailable)
# ---------------------------------------------------------------------------

_INLINE_RUBRIC_SUMMARIES: dict[int, str] = {
    1: (
        "You have absolutely no coding background. Describe what you want in plain everyday "
        "language. Do not use technical terms like function, variable, loop, or error. Focus "
        "on what the program should DO, not how. Trust the developer completely. React only to "
        "visible results. Sound casual and non-technical."
    ),
    2: (
        "You have low coding literacy. You can describe expected behavior with concrete examples "
        "(if I give it 5 it should give 10). You may paste error messages you see on screen but "
        "cannot interpret them. Avoid jargon. Describe what you observed, not technical causes."
    ),
    3: (
        "You are a beginner coder. You can read simple function names and basic syntax. You run "
        "the code manually and describe what happened. Your feedback is mostly symptom-driven. "
        "You may ask for specific simple fixes and name the function you think is wrong."
    ),
    4: (
        "You are an intermediate coder. You describe inputs/outputs and edge cases. You can "
        "inspect code shallowly and mention likely problem areas. You reference function names "
        "and can read error messages. You ask about correctness and sometimes about code quality."
    ),
    5: (
        "You are an advanced coder. You write structured requirements and test the code carefully. "
        "You have lower trust and more verification mindset. You mention architecture, "
        "maintainability, and specific implementation concerns. You cite function names and lines."
    ),
    6: (
        "You are an expert developer. You give precise specifications and do systematic debugging. "
        "You do thorough code review. You explicitly care about correctness, architecture, testing, "
        "performance, and security. You cite specific lines, edge cases, and design decisions."
    ),
}


def _rubric_for_level(level: int) -> str:
    """Load the level rubric; fall back to inline summary if file unavailable."""
    text = _load_rubric_text(level)
    if text:
        # Condense the markdown file to its most actionable parts for the prompt
        # Keep the "Who This User Is" and "How They Frame" sections
        lines = text.splitlines()
        keep = []
        in_section = False
        for line in lines:
            if line.startswith("## Who This User Is") or line.startswith("## How They Frame") or line.startswith("## How They Report") or line.startswith("## Trust") or line.startswith("## Vocabulary") or line.startswith("## What They Care"):
                in_section = True
            elif line.startswith("## ") and in_section:
                in_section = False
            if in_section and not line.startswith("## "):
                stripped = line.strip()
                if stripped:
                    keep.append(stripped)
        if keep:
            return " ".join(keep)[:800]
    return _INLINE_RUBRIC_SUMMARIES.get(level, _INLINE_RUBRIC_SUMMARIES[3])


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip() + " ..."


def build_initial_user_prompt(
    *,
    chain: dict[str, Any],
    knowledge_level: int,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for round 1 user message."""
    level_label = LEVEL_LABELS.get(knowledge_level, "intermediate coder")
    rubric = _rubric_for_level(knowledge_level)

    system_prompt = (
        f"You are roleplaying as a human who wants to commission a software program. "
        f"Your knowledge level: {level_label}.\n"
        f"Behavioral guide:\n{rubric}\n\n"
        "Rules:\n"
        "- Sound like a real human, not a specification or benchmark.\n"
        "- Write in first person.\n"
        "- Do NOT use numbered lists or headers unless your level is 5 or 6.\n"
        "- Do NOT reveal you are an AI or simulator.\n"
        "- Do NOT describe the rubric or your level explicitly.\n"
        f"- Keep the message under about {150 if knowledge_level >= 5 else 80} words."
    )

    task = str(chain.get("question", "")).strip()
    prompt = (
        f"You want the developer to write a program for this task:\n\n{task}\n\n"
        f"Write your first message to the developer describing what you want. "
        f"Be natural. Sound like a {level_label}."
    )
    return system_prompt, prompt


def build_followup_user_prompt(
    *,
    chain: dict[str, Any],
    knowledge_level: int,
    round_index: int,
    tester_public_feedback: str,
    prev_code_artifact: str,
    memory_nodes: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for follow-up user messages (round >= 2)."""
    level_label = LEVEL_LABELS.get(knowledge_level, "intermediate coder")
    rubric = _rubric_for_level(knowledge_level)

    system_prompt = (
        f"You are roleplaying as a human who commissioned a software program. "
        f"Your knowledge level: {level_label}.\n"
        f"Behavioral guide:\n{rubric}\n\n"
        "Rules:\n"
        "- Sound like a real human, not a specification or benchmark.\n"
        "- Write in first person.\n"
        "- Translate the tester's findings into YOUR OWN WORDS based on your knowledge level.\n"
        "- Low-skill users should describe symptoms in plain English.\n"
        "- High-skill users can cite specific failures or propose fixes.\n"
        "- Do NOT dump all the tester findings verbatim.\n"
        "- Do NOT reveal you are an AI or simulator.\n"
        f"- Keep the message under about {200 if knowledge_level >= 5 else 100} words."
    )

    lines: list[str] = [
        f"Original task you requested:",
        str(chain.get("question", "")).strip()[:400],
        "",
        f"The developer has given you code in round {round_index - 1}.",
        "A tester tested it and found the following issues:",
        "",
        str(tester_public_feedback).strip() if tester_public_feedback else "(tester found no specific issues)",
        "",
    ]

    # Show code to higher-knowledge users
    if knowledge_level >= 4 and prev_code_artifact:
        lines += [
            "The developer's latest code (you may reference it at your level):",
            f"```python\n{prev_code_artifact[:500]}\n```",
            "",
        ]

    # Show recent conversation memory
    if memory_nodes:
        lines += ["Recent conversation memory:"]
        for mnode in memory_nodes[-(2 if knowledge_level <= 3 else 3):]:
            msg = str(mnode.get("user_message", "")).strip()
            if msg:
                lines.append(
                    f"You said (round {mnode['round']}): {_truncate_words(msg, 25)}"
                )
        lines.append("")

    lines += [
        f"Write your next message to the developer as a {level_label}.",
        "Be natural and use vocabulary consistent with your level.",
    ]
    prompt = "\n".join(lines)
    return system_prompt, prompt


# ---------------------------------------------------------------------------
# Fallback message generation (no API call needed)
# ---------------------------------------------------------------------------

def _fallback_initial_message(chain: dict[str, Any], knowledge_level: int) -> str:
    task = str(chain.get("question", "")).strip()[:180]
    if knowledge_level <= 2:
        return f"Hi, I want a program that does this: {task}. Can you make that for me?"
    if knowledge_level <= 4:
        return f"Please write code for this: {task}"
    return f"I need an implementation for the following: {task}"


def _fallback_followup_message(
    chain: dict[str, Any],
    knowledge_level: int,
    tester_public_feedback: str,
) -> str:
    feedback_snippet = _truncate_words(str(tester_public_feedback or "").strip(), 20)
    if knowledge_level <= 2:
        return f"It's still not working right. {feedback_snippet or 'Can you fix it please?'}"
    if knowledge_level <= 4:
        return f"There are still issues: {feedback_snippet or 'please check and fix.'}"
    return (
        f"The following problems remain from the tester: {feedback_snippet or 'please review.'}"
        + " Please address them."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

UserInferenceFn = Callable[[str, str], tuple[str, str]]


def generate_user_message(
    *,
    chain: dict[str, Any],
    round_index: int,
    knowledge_level: int,
    user_inference_fn: UserInferenceFn | None,
    user_config: VibeCodingUserConfig,
    logger,
) -> tuple[str, str, str]:
    """Generate a user message for the given round.

    Returns:
        (message_text, prompt_used, error_text)
    """
    if round_index == 1:
        system_prompt, prompt = build_initial_user_prompt(
            chain=chain,
            knowledge_level=knowledge_level,
        )
        fallback_fn = lambda: _fallback_initial_message(chain, knowledge_level)
    else:
        prev_nodes = chain.get("nodes", [])
        prev_node = prev_nodes[-1] if prev_nodes else {}
        tester_public_feedback = str(prev_node.get("tester_public_feedback", "")).strip()
        prev_code_artifact = str(prev_node.get("code_artifact", "")).strip()
        window = max(1, user_config.memory_window)
        memory_nodes = prev_nodes[:-1][-window:] if len(prev_nodes) > 1 else []

        system_prompt, prompt = build_followup_user_prompt(
            chain=chain,
            knowledge_level=knowledge_level,
            round_index=round_index,
            tester_public_feedback=tester_public_feedback,
            prev_code_artifact=prev_code_artifact,
            memory_nodes=memory_nodes,
        )
        fallback_fn = lambda: _fallback_followup_message(chain, knowledge_level, tester_public_feedback)

    raw_text = ""
    error_text = ""
    if user_inference_fn is not None:
        try:
            raw_text, error_text = user_inference_fn(prompt, system_prompt)
        except Exception as exc:
            error_text = f"{exc.__class__.__name__}: {exc}"

    if error_text and logger is not None:
        logger.warning(
            "User simulator error on chain %s round %s: %s",
            chain.get("chain_id", "?"),
            round_index,
            error_text,
        )

    message = raw_text.strip()
    if not message:
        message = fallback_fn()

    # Strip any accidental "User:" prefix
    message = re.sub(r"^user\s*:\s*", "", message, flags=re.IGNORECASE).strip()

    return message, prompt, error_text
