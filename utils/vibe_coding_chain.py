"""Vibe-coding three-role experiment orchestration.

Implements the chain-building, round loop, saving, resume, and metrics logic
for the vibe-coding experiment family.  Reuses shared infrastructure from
multiturn_chain.py where possible.

Each chain = one coding task + one developer model + one user knowledge level.
Flow per round:
  1. User simulator  -> produces user message to developer
  2. Developer model -> produces code (no system prompt, no formatting restriction)
  3. Tester model    -> generates/runs public+hidden tests, returns public feedback

Marking: chain is marked_out after `consecutive_hidden_pass >= consecutive_k` rounds of
100% hidden pass rate (configurable).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from utils.eval_code import extract_python_code
from utils.multiturn_chain import (
    append_jsonl,
    create_experiment_dir,
    ensure_dir,
    load_resume_snapshot,
    read_json,
    save_chain_states,
    save_dataset_snapshot,
    slugify,
    text_similarity,
    utc_now,
    write_json,
)
from utils.vibe_coding_tester import (
    TesterOutput,
    VibeCodingTesterConfig,
    run_tester,
    tester_output_to_dict,
)
from utils.vibe_coding_user_sim import (
    VibeCodingUserConfig,
    generate_user_message,
    sample_knowledge_level,
)

# ---------------------------------------------------------------------------
# Type aliases for inference callables
# ---------------------------------------------------------------------------

DeveloperInferenceFn = Callable[[dict[str, Any], str, int], tuple[str, str]]
UserInferenceFn = Callable[[str, str], tuple[str, str]]
TesterInferenceFn = Callable[[str, str], tuple[str, str]]

# ---------------------------------------------------------------------------
# Chain building
# ---------------------------------------------------------------------------

def _vibe_chain_id(sample_id: str, developer_profile: str, knowledge_level: int) -> str:
    base = f"{sample_id}::vibe::{developer_profile}::level{knowledge_level}"
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", sample_id)
    return f"{clean}__vibe__{slugify(developer_profile)}__l{knowledge_level}__{digest}"


def build_vibe_chains(
    samples: list[dict[str, Any]],
    developer_profiles: list[dict[str, str]],
    level_weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build vibe-coding chains.

    Each chain = (coding sample, developer model profile, sampled knowledge level).
    Only code-domain samples are used.

    Args:
        samples: benchmark samples from load_benchmark_samples()
        developer_profiles: list of dicts with "profile_name" and "model" keys
        level_weights: optional override for knowledge level distribution
    """
    chains: list[dict[str, Any]] = []
    for sample in samples:
        # Only code-domain tasks are relevant for vibe coding
        if sample.get("task_type") not in {"code", "swebench"} and sample.get("domain") not in {"code"}:
            continue
        for profile in developer_profiles:
            profile_name = profile["profile_name"]
            developer_model = profile.get("model", profile.get("code", ""))
            # Build a temporary chain_id seed for level sampling
            seed_id = f"{sample['sample_id']}::{profile_name}"
            knowledge_level = sample_knowledge_level(seed_id, level_weights)
            chain_id = _vibe_chain_id(sample["sample_id"], profile_name, knowledge_level)
            chain: dict[str, Any] = {
                "chain_id": chain_id,
                "sample_id": sample["sample_id"],
                "dataset_name": sample["dataset_name"],
                "domain": sample.get("domain", "code"),
                "task_type": sample.get("task_type", "code"),
                "question": sample["question"],
                "reference_answer": sample.get("reference_answer", ""),
                "metadata": sample.get("metadata", {}),
                "developer_model": developer_model,
                "developer_profile": profile_name,
                "user_knowledge_level": knowledge_level,
                "status": "active",
                "marked_out_round": None,
                "consecutive_hidden_pass": 0,
                "nodes": [],
            }
            chains.append(chain)
    chains.sort(key=lambda c: c["chain_id"])
    return chains


# ---------------------------------------------------------------------------
# Developer prompt construction
# ---------------------------------------------------------------------------

def build_developer_prompt(chain: dict[str, Any], round_index: int) -> str:
    """Build the developer prompt for the given round.

    Round 1: just the user's first message.
    Round 2+: conversation history with user messages and developer's previous code.

    No system prompt is used for the developer (per spec).
    """
    nodes = chain.get("nodes", [])

    if round_index == 1 or not nodes:
        # First round: user describes the task
        user_message = str(chain.get("_pending_user_message", "")).strip()
        if not user_message:
            user_message = str(chain.get("question", "")).strip()
        return user_message

    # Build conversation history
    lines: list[str] = []
    for node in nodes:
        user_msg = str(node.get("user_message", "")).strip()
        if user_msg:
            lines.append(f"User: {user_msg}")
        dev_code = str(node.get("code_artifact", "")).strip()
        dev_raw = str(node.get("developer_raw_output", "")).strip()
        if dev_code:
            lines.append(f"Developer:\n```python\n{dev_code}\n```")
        elif dev_raw:
            lines.append(f"Developer: {dev_raw[:600]}")
        # Optionally include public feedback so developer sees it in context
        public_fb = str(node.get("tester_public_feedback", "")).strip()
        if public_fb:
            lines.append(f"[Test feedback for the user]: {public_fb}")

    # Append current user message
    current_user_msg = str(chain.get("_pending_user_message", "")).strip()
    if current_user_msg:
        lines.append(f"User: {current_user_msg}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Code extraction from developer output
# ---------------------------------------------------------------------------

def extract_code_from_developer_output(raw_output: str) -> str:
    """Extract Python code from the developer's raw output."""
    return extract_python_code(raw_output)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: list[float]) -> float | None:
    return _mean(values) if values else None


def compute_vibe_round_metrics(
    chains: list[dict[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    """Compute global round-level metrics."""
    nodes = [
        c["nodes"][-1]
        for c in chains
        if c.get("nodes") and c["nodes"][-1].get("round") == round_index
    ]
    if not nodes:
        return {"round": round_index, "num_chains": 0}

    hidden_rates = [float(n["hidden_pass_rate"]) for n in nodes if n.get("hidden_pass_rate") is not None]
    public_rates = [float(n["public_pass_rate"]) for n in nodes if n.get("public_pass_rate") is not None]
    code_sims = [float(n["code_similarity_prev"]) for n in nodes if n.get("code_similarity_prev") is not None]
    regressions = [int(n.get("hidden_regression_count", 0)) for n in nodes]
    regressions_accum = [int(n.get("hidden_regression_accum", 0)) for n in nodes]
    hidden_counts = [int(n.get("hidden_test_count", 0)) for n in nodes]
    public_counts = [int(n.get("public_test_count", 0)) for n in nodes]

    marked = sum(1 for c in chains if c["status"] == "marked_out")
    active = len(chains) - marked

    return {
        "round": round_index,
        "num_chains": len(chains),
        "mean_hidden_pass_rate": _mean_optional(hidden_rates),
        "mean_public_pass_rate": _mean_optional(public_rates),
        "mean_code_similarity_prev": _mean_optional(code_sims),
        "total_hidden_regression_count": int(sum(regressions)),
        "total_hidden_regression_accum": int(sum(regressions_accum)),
        "mean_hidden_test_count": _mean_optional([float(v) for v in hidden_counts]),
        "mean_public_test_count": _mean_optional([float(v) for v in public_counts]),
        "active_chains": active,
        "marked_out_chains": marked,
        "api_failure_count": int(sum(1 for n in nodes if n.get("developer_error"))),
        "exec_failure_count": int(sum(1 for n in nodes if n.get("tester_error"))),
    }


def _condition_vibe_metrics_rows(
    chains: list[dict[str, Any]],
    round_index: int,
) -> list[dict[str, Any]]:
    """Per-(dataset, developer_model, knowledge_level) metrics."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    status_map: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"active": 0, "marked_out": 0}
    )

    for chain in chains:
        key = (chain["dataset_name"], chain["developer_model"], chain["user_knowledge_level"])
        if chain["status"] == "marked_out":
            status_map[key]["marked_out"] += 1
        else:
            status_map[key]["active"] += 1
        if not chain.get("nodes"):
            continue
        node = chain["nodes"][-1]
        if int(node.get("round", 0)) != round_index:
            continue
        grouped[key].append(node)

    rows: list[dict[str, Any]] = []
    for key, nodes in sorted(grouped.items(), key=lambda x: x[0]):
        dataset_name, dev_model, klevel = key
        hidden_rates = [float(n["hidden_pass_rate"]) for n in nodes if n.get("hidden_pass_rate") is not None]
        public_rates = [float(n["public_pass_rate"]) for n in nodes if n.get("public_pass_rate") is not None]
        rows.append({
            "round": round_index,
            "dataset_name": dataset_name,
            "developer_model": dev_model,
            "knowledge_level": klevel,
            "num_chains": len(nodes),
            "mean_hidden_pass_rate": _mean_optional(hidden_rates),
            "mean_public_pass_rate": _mean_optional(public_rates),
            "active_chains": status_map[key]["active"],
            "marked_out_chains": status_map[key]["marked_out"],
        })
    return rows


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _vibe_condition_dir(exp_dir: Path, chain: dict[str, Any]) -> Path:
    return (
        exp_dir
        / "chain"
        / slugify(chain["dataset_name"])
        / slugify(chain["developer_model"])
        / f"level_{chain['user_knowledge_level']}"
    )


def save_vibe_round_outputs(
    exp_dir: Path,
    chains: list[dict[str, Any]],
    round_index: int,
    round_metrics: dict[str, Any],
    *,
    save_artifacts: bool = True,
) -> list[dict[str, Any]]:
    """Save all per-round outputs and return condition metrics rows."""
    round_file = exp_dir / "rounds" / f"round_{round_index:03d}.jsonl"
    condition_metrics = _condition_vibe_metrics_rows(chains, round_index)

    for chain in chains:
        if not chain.get("nodes"):
            continue
        node = chain["nodes"][-1]
        if int(node.get("round", 0)) != round_index:
            continue

        payload: dict[str, Any] = {
            "chain_id": chain["chain_id"],
            "sample_id": chain["sample_id"],
            "dataset_name": chain["dataset_name"],
            "task_type": chain["task_type"],
            "developer_model": chain["developer_model"],
            "developer_profile": chain["developer_profile"],
            "user_knowledge_level": chain["user_knowledge_level"],
            "node": node,
        }
        append_jsonl(round_file, payload)

        cdir = _vibe_condition_dir(exp_dir, chain)
        qid = slugify(str(chain["sample_id"]).split("::")[-1])
        append_jsonl(cdir / f"data_{qid}.jsonl", payload)

        # Save code artifact as a separate file for easy inspection
        if save_artifacts:
            artifact_code = str(node.get("code_artifact", "")).strip()
            if artifact_code:
                artifact_dir = exp_dir / "artifacts" / slugify(chain["chain_id"])
                ensure_dir(artifact_dir)
                (artifact_dir / f"round_{round_index:03d}.py").write_text(
                    artifact_code, encoding="utf-8"
                )

        # Save user simulator trace
        user_record = {
            "round": round_index,
            "chain_id": chain["chain_id"],
            "sample_id": chain["sample_id"],
            "dataset_name": chain["dataset_name"],
            "developer_model": chain["developer_model"],
            "knowledge_level": chain["user_knowledge_level"],
            "user_message": node.get("user_message", ""),
            "user_prompt": node.get("user_prompt", ""),
            "user_error": node.get("user_error", ""),
        }
        append_jsonl(exp_dir / "user-simulator-records" / "all_user_interactions.jsonl", user_record)
        append_jsonl(
            exp_dir / "user-simulator-records" / slugify(str(chain["dataset_name"])) / "user_interactions.jsonl",
            user_record,
        )

        # Save tester trace
        tester_record = {
            "round": round_index,
            "chain_id": chain["chain_id"],
            "sample_id": chain["sample_id"],
            "dataset_name": chain["dataset_name"],
            "developer_model": chain["developer_model"],
            "knowledge_level": chain["user_knowledge_level"],
            "tester_model": node.get("tester_model", ""),
            "tester_prompt": node.get("tester_prompt", "")[:1000],
            "tester_raw_output": node.get("tester_raw_output", "")[:2000],
            "tester_error": node.get("tester_error", ""),
            "public_tests": node.get("public_tests", []),
            "public_results": node.get("public_results", []),
            "hidden_tests": node.get("hidden_tests", []),
            "hidden_results": node.get("hidden_results", []),
            "public_pass_rate": node.get("public_pass_rate"),
            "hidden_pass_rate": node.get("hidden_pass_rate"),
            "public_feedback": node.get("tester_public_feedback", ""),
        }
        append_jsonl(exp_dir / "tester-records" / "all_tester_interactions.jsonl", tester_record)
        append_jsonl(
            exp_dir / "tester-records" / slugify(str(chain["dataset_name"])) / "tester_interactions.jsonl",
            tester_record,
        )

    # Condition metrics
    for row in condition_metrics:
        append_jsonl(
            _vibe_condition_dir(
                exp_dir,
                {
                    "dataset_name": row["dataset_name"],
                    "developer_model": row["developer_model"],
                    "user_knowledge_level": row["knowledge_level"],
                },
            )
            / "metrics_round.jsonl",
            row,
        )
        append_jsonl(exp_dir / "rounds" / "vibe_condition_metrics_round.jsonl", row)

    append_jsonl(exp_dir / "rounds" / "round_metrics.jsonl", round_metrics)

    # Snapshot for resume
    write_json(
        exp_dir / "snapshots" / f"round_{round_index:03d}.json",
        {"round": round_index, "round_metrics": round_metrics, "chains": chains},
    )

    return condition_metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_vibe_summaries(
    chains: list[dict[str, Any]],
    exp_dir: Path,
    max_rounds: int,
) -> None:
    """Write end-of-run summary dataframes."""
    summary_dir = exp_dir / "summary"
    ensure_dir(summary_dir)

    detailed_rows: list[dict[str, Any]] = []
    training_traces: list[dict[str, Any]] = []

    for chain in chains:
        for node in chain.get("nodes", []):
            row = {
                "max_rounds": max_rounds,
                "dataset_name": chain["dataset_name"],
                "developer_model": chain["developer_model"],
                "developer_profile": chain["developer_profile"],
                "knowledge_level": chain["user_knowledge_level"],
                "sample_id": chain["sample_id"],
                "chain_id": chain["chain_id"],
                "round": node.get("round"),
                "hidden_pass_rate": node.get("hidden_pass_rate"),
                "public_pass_rate": node.get("public_pass_rate"),
                "hidden_test_count": node.get("hidden_test_count"),
                "public_test_count": node.get("public_test_count"),
                "code_similarity_prev": node.get("code_similarity_prev"),
                "hidden_regression_count": node.get("hidden_regression_count"),
                "hidden_regression_accum": node.get("hidden_regression_accum"),
                "marked_out": int(bool(node.get("marked_out", False))),
                "consecutive_hidden_pass": node.get("consecutive_hidden_pass"),
                "skipped": int(bool(node.get("skipped", False))),
                "developer_error": int(bool(node.get("developer_error"))),
                "tester_error": int(bool(node.get("tester_error"))),
            }
            detailed_rows.append(row)

            # SFT-style training trace (for developer model training)
            if not node.get("skipped") and node.get("user_message") and node.get("developer_raw_output"):
                training_traces.append({
                    "chain_id": chain["chain_id"],
                    "sample_id": chain["sample_id"],
                    "dataset_name": chain["dataset_name"],
                    "knowledge_level": chain["user_knowledge_level"],
                    "round": node.get("round"),
                    "developer_model": chain["developer_model"],
                    "prompt": node.get("developer_prompt", ""),
                    "response": node.get("developer_raw_output", ""),
                    "code_artifact": node.get("code_artifact", ""),
                    "hidden_pass_rate": node.get("hidden_pass_rate"),
                    "is_good_response": node.get("hidden_pass_rate", 0.0) >= 0.8,
                })

    # Detailed dataframe
    dj = summary_dir / "summary_vibe_dataframe.jsonl"
    if dj.exists():
        dj.unlink()
    for row in detailed_rows:
        append_jsonl(dj, row)
    _write_csv(summary_dir / "summary_vibe_dataframe.csv", detailed_rows)

    # Aggregated by (dataset, developer_model, level, round)
    agg_map: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed_rows:
        key = (row["dataset_name"], row["developer_model"], row["knowledge_level"], row["round"])
        agg_map[key].append(row)

    agg_rows: list[dict[str, Any]] = []
    for key, rows in sorted(agg_map.items(), key=lambda x: x[0]):
        dataset_name, dev_model, klevel, round_index = key
        hr = [float(r["hidden_pass_rate"]) for r in rows if r.get("hidden_pass_rate") is not None]
        pr = [float(r["public_pass_rate"]) for r in rows if r.get("public_pass_rate") is not None]
        agg_rows.append({
            "max_rounds": max_rounds,
            "dataset_name": dataset_name,
            "developer_model": dev_model,
            "knowledge_level": klevel,
            "round": round_index,
            "num_chains": len(rows),
            "mean_hidden_pass_rate": _mean_optional(hr),
            "mean_public_pass_rate": _mean_optional(pr),
            "marked_out_chains": int(sum(int(r["marked_out"]) for r in rows)),
        })

    rj = summary_dir / "summary_vibe_round_by_condition.jsonl"
    if rj.exists():
        rj.unlink()
    for row in agg_rows:
        append_jsonl(rj, row)
    _write_csv(summary_dir / "summary_vibe_round_by_condition.csv", agg_rows)

    # Training traces
    tj = summary_dir / "training_traces.jsonl"
    if tj.exists():
        tj.unlink()
    for trace in training_traces:
        append_jsonl(tj, trace)


# ---------------------------------------------------------------------------
# Carry-forward for marked chains
# ---------------------------------------------------------------------------

def _carry_forward_vibe_node(chain: dict[str, Any], round_index: int) -> dict[str, Any]:
    previous = chain["nodes"][-1] if chain["nodes"] else {}
    prev_accum = int(previous.get("hidden_regression_accum", 0))
    return {
        "round": round_index,
        "timestamp": utc_now(),
        "skipped": True,
        "skip_reason": "marked_out",
        "user_message": previous.get("user_message", ""),
        "user_prompt": "",
        "user_error": "",
        "developer_prompt": "",
        "developer_raw_output": previous.get("developer_raw_output", ""),
        "code_artifact": previous.get("code_artifact", ""),
        "developer_error": "",
        "tester_public_feedback": previous.get("tester_public_feedback", ""),
        "tester_raw_output": "",
        "tester_prompt": "",
        "tester_error": "",
        "public_tests": previous.get("public_tests", []),
        "public_results": previous.get("public_results", []),
        "public_pass_rate": previous.get("public_pass_rate", 0.0),
        "public_test_count": previous.get("public_test_count", 0),
        "hidden_tests": previous.get("hidden_tests", []),
        "hidden_results": previous.get("hidden_results", []),
        "hidden_pass_rate": previous.get("hidden_pass_rate", 0.0),
        "hidden_test_count": previous.get("hidden_test_count", 0),
        "code_similarity_prev": 1.0,
        "hidden_regression_count": 0,
        "hidden_regression_accum": prev_accum,
        "consecutive_hidden_pass": chain.get("consecutive_hidden_pass", 0),
        "marked_out": True,
        "developer_model": chain["developer_model"],
        "user_model": previous.get("user_model", ""),
        "tester_model": previous.get("tester_model", ""),
    }


def _apply_vibe_node(
    chain: dict[str, Any],
    node: dict[str, Any],
    consecutive_k: int,
) -> None:
    """Update chain state and append node."""
    prev_node = chain["nodes"][-1] if chain["nodes"] else None
    prev_accum = int((prev_node.get("hidden_regression_accum", 0) if prev_node else 0))

    # Regression: hidden tests that previously passed now fail
    if prev_node and not prev_node.get("skipped"):
        prev_hidden_results = prev_node.get("hidden_results", [])
        curr_hidden_results = node.get("hidden_results", [])
        prev_pass_codes = {r["code"] for r in prev_hidden_results if r.get("passed")}
        curr_fail_codes = {r["code"] for r in curr_hidden_results if not r.get("passed")}
        regression_count = len(prev_pass_codes & curr_fail_codes)
    else:
        regression_count = 0

    node["hidden_regression_count"] = regression_count
    node["hidden_regression_accum"] = prev_accum + regression_count

    # Code similarity
    prev_code = str(prev_node.get("code_artifact", "") if prev_node else "")
    curr_code = str(node.get("code_artifact", ""))
    sim = text_similarity(prev_code, curr_code) if prev_node else None
    node["code_similarity_prev"] = sim

    # Consecutive hidden pass tracking
    hidden_pass_rate = float(node.get("hidden_pass_rate", 0.0))
    if hidden_pass_rate >= 1.0:
        chain["consecutive_hidden_pass"] = int(chain.get("consecutive_hidden_pass", 0)) + 1
    else:
        chain["consecutive_hidden_pass"] = 0
    node["consecutive_hidden_pass"] = chain["consecutive_hidden_pass"]

    # Mark out
    if chain["status"] == "active" and chain["consecutive_hidden_pass"] >= consecutive_k:
        chain["status"] = "marked_out"
        chain["marked_out_round"] = node["round"]
    node["marked_out"] = chain["status"] == "marked_out"

    chain["nodes"].append(node)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

@dataclass
class VibeCodingExperimentConfig:
    max_rounds: int = 5
    consecutive_k: int = 2
    workers: int = 4
    test_timeout: int = 15
    seed: int = 42
    dry_run: bool = False


def run_vibe_coding_experiment(
    *,
    chains: list[dict[str, Any]],
    exp_dir: Path,
    developer_inference_fn: DeveloperInferenceFn,
    user_inference_fn: UserInferenceFn | None,
    tester_inference_fn: TesterInferenceFn | None,
    user_config: VibeCodingUserConfig,
    tester_config: VibeCodingTesterConfig,
    exp_config: VibeCodingExperimentConfig,
    logger,
    start_round: int = 1,
    round_hook: Callable[[int, dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> None:
    """Main vibe-coding experiment loop.

    Args:
        chains: list of vibe-coding chains (from build_vibe_chains)
        exp_dir: experiment output directory
        developer_inference_fn: (chain, prompt, round_index) -> (raw_output, error)
        user_inference_fn: (prompt, system_prompt) -> (text, error)  or None for toy mode
        tester_inference_fn: (prompt, system_prompt) -> (text, error)  or None for toy mode
        user_config: VibeCodingUserConfig
        tester_config: VibeCodingTesterConfig
        exp_config: VibeCodingExperimentConfig
        logger: Python logger instance
        start_round: first round to run (1 for fresh, N+1 when resuming from round N)
        round_hook: optional callback(round_index, round_metrics, condition_metrics)
    """
    save_chain_states(chains, exp_dir / "chains")

    for round_index in range(start_round, exp_config.max_rounds + 1):
        logger.info("Vibe coding round %s / %s  |  chains=%s", round_index, exp_config.max_rounds, len(chains))

        active_chains = [c for c in chains if c["status"] == "active"]
        skipped_chains = [c for c in chains if c["status"] == "marked_out"]

        # --- Step 1: User simulator generates messages for active chains ---
        logger.info("  Step 1: user simulator  (%s active chains)", len(active_chains))

        def _gen_user(chain: dict[str, Any]) -> tuple[str, str, str]:
            return generate_user_message(
                chain=chain,
                round_index=round_index,
                knowledge_level=chain["user_knowledge_level"],
                user_inference_fn=user_inference_fn,
                user_config=user_config,
                logger=logger,
            )

        user_results: dict[str, tuple[str, str, str]] = {}
        if exp_config.workers > 1 and len(active_chains) > 1:
            with ThreadPoolExecutor(max_workers=exp_config.workers) as pool:
                fmap = {pool.submit(_gen_user, c): c["chain_id"] for c in active_chains}
                for fut in as_completed(fmap):
                    cid = fmap[fut]
                    try:
                        user_results[cid] = fut.result()
                    except Exception as exc:
                        user_results[cid] = ("", "", str(exc))
        else:
            for chain in active_chains:
                try:
                    user_results[chain["chain_id"]] = _gen_user(chain)
                except Exception as exc:
                    user_results[chain["chain_id"]] = ("", "", str(exc))

        # Store pending user messages on chains for prompt building
        for chain in active_chains:
            msg, _prompt, _err = user_results.get(chain["chain_id"], ("", "", ""))
            chain["_pending_user_message"] = msg

        # --- Step 2: Developer generates code ---
        logger.info("  Step 2: developer inference")

        dev_results: dict[str, tuple[str, str, str]] = {}  # chain_id -> (prompt, raw_out, error)
        if exp_config.workers > 1 and len(active_chains) > 1:
            with ThreadPoolExecutor(max_workers=exp_config.workers) as pool:
                fmap = {}
                for chain in active_chains:
                    prompt_text = build_developer_prompt(chain, round_index)
                    fut = pool.submit(developer_inference_fn if not exp_config.dry_run else
                                      lambda c, p, r: (f"def solution():\n    pass  # round {r}", ""),
                                      chain, prompt_text, round_index)
                    fmap[fut] = (chain["chain_id"], prompt_text)
                for fut in as_completed(fmap):
                    cid, prompt_text = fmap[fut]
                    try:
                        raw_out, err = fut.result()
                    except Exception as exc:
                        raw_out, err = "", str(exc)
                    dev_results[cid] = (prompt_text, raw_out, err)
        else:
            for chain in active_chains:
                prompt_text = build_developer_prompt(chain, round_index)
                if exp_config.dry_run:
                    raw_out, err = f"def solution():\n    pass  # round {round_index}", ""
                else:
                    try:
                        raw_out, err = developer_inference_fn(chain, prompt_text, round_index)
                    except Exception as exc:
                        raw_out, err = "", str(exc)
                dev_results[chain["chain_id"]] = (prompt_text, raw_out, err)

        # Pre-extract code artifacts (avoid redundant extraction in tester + node assembly)
        code_artifacts: dict[str, str] = {}
        for chain in active_chains:
            _pt, raw_out, _err = dev_results.get(chain["chain_id"], ("", "", ""))
            code_artifacts[chain["chain_id"]] = extract_code_from_developer_output(raw_out)

        # --- Step 3: Tester evaluates each chain ---
        logger.info("  Step 3: tester evaluation")

        def _run_chain_tester(chain: dict[str, Any]) -> TesterOutput:
            code_artifact = code_artifacts.get(chain["chain_id"], "")
            user_messages = [
                str(n.get("user_message", "")) for n in chain.get("nodes", [])
            ] + [str(chain.get("_pending_user_message", ""))]

            return run_tester(
                chain=chain,
                code_artifact=code_artifact,
                round_index=round_index,
                user_messages=[m for m in user_messages if m.strip()],
                tester_inference_fn=tester_inference_fn,
                tester_config=tester_config,
                logger=logger,
            )

        tester_results: dict[str, TesterOutput] = {}
        if exp_config.workers > 1 and len(active_chains) > 1:
            with ThreadPoolExecutor(max_workers=exp_config.workers) as pool:
                fmap = {pool.submit(_run_chain_tester, c): c["chain_id"] for c in active_chains}
                for fut in as_completed(fmap):
                    cid = fmap[fut]
                    try:
                        tester_results[cid] = fut.result()
                    except Exception as exc:
                        logger.warning("Tester future failed for chain %s: %s", cid, exc)
                        tester_results[cid] = TesterOutput(tester_error=str(exc))
        else:
            for chain in active_chains:
                try:
                    tester_results[chain["chain_id"]] = _run_chain_tester(chain)
                except Exception as exc:
                    logger.warning("Tester failed for chain %s: %s", chain["chain_id"], exc)
                    tester_results[chain["chain_id"]] = TesterOutput(tester_error=str(exc))

        # --- Assemble nodes ---
        for chain in chains:
            if chain["status"] == "marked_out":
                node = _carry_forward_vibe_node(chain, round_index)
                chain["nodes"].append(node)
                continue

            user_msg, user_prompt_text, user_err = user_results.get(chain["chain_id"], ("", "", ""))
            dev_prompt_text, dev_raw_output, dev_error = dev_results.get(chain["chain_id"], ("", "", ""))
            code_artifact = code_artifacts.get(chain["chain_id"], "")
            tester_out = tester_results.get(chain["chain_id"], TesterOutput())
            tout_dict = tester_output_to_dict(tester_out)

            node: dict[str, Any] = {
                "round": round_index,
                "timestamp": utc_now(),
                "skipped": False,
                "skip_reason": "",
                # User turn
                "user_message": user_msg,
                "user_prompt": user_prompt_text[:1500],
                "user_error": user_err,
                # Developer turn
                "developer_prompt": dev_prompt_text[:2000],
                "developer_raw_output": dev_raw_output,
                "code_artifact": code_artifact,
                "developer_error": dev_error,
                # Tester turn
                "tester_public_feedback": tester_out.public_feedback,
                "tester_raw_output": tester_out.tester_raw_output[:2000],
                "tester_prompt": tester_out.tester_prompt[:1500],
                "tester_error": tester_out.tester_error,
                "public_tests": tout_dict["public_tests"],
                "public_results": tout_dict["public_results"],
                "public_pass_rate": tester_out.public_pass_rate,
                "public_test_count": tester_out.public_test_count,
                "hidden_tests": tout_dict["hidden_tests"],
                "hidden_results": tout_dict["hidden_results"],
                "hidden_pass_rate": tester_out.hidden_pass_rate,
                "hidden_test_count": tester_out.hidden_test_count,
                # These are filled by _apply_vibe_node
                "code_similarity_prev": None,
                "hidden_regression_count": 0,
                "hidden_regression_accum": 0,
                "consecutive_hidden_pass": 0,
                "marked_out": False,
                # Model metadata
                "developer_model": chain["developer_model"],
                "user_model": user_config.model,
                "tester_model": tester_config.model,
            }

            _apply_vibe_node(chain, node, exp_config.consecutive_k)

        # Clean up transient state before saving
        for chain in chains:
            chain.pop("_pending_user_message", None)

        # --- Save round outputs ---
        round_metrics = compute_vibe_round_metrics(chains, round_index)
        condition_metrics = save_vibe_round_outputs(
            exp_dir,
            chains,
            round_index,
            round_metrics,
        )
        save_chain_states(chains, exp_dir / "chains")

        logger.info(
            "  Round %s done: hidden_pass_rate=%.3f  active=%s  marked=%s",
            round_index,
            round_metrics.get("mean_hidden_pass_rate") or 0.0,
            round_metrics.get("active_chains", 0),
            round_metrics.get("marked_out_chains", 0),
        )

        if round_hook is not None:
            try:
                round_hook(round_index, round_metrics, condition_metrics)
            except Exception as exc:
                logger.warning("Round hook failed at round %s: %s", round_index, exc)

    # End-of-run summaries
    save_vibe_summaries(chains, exp_dir, exp_config.max_rounds)
    logger.info("Vibe coding experiment finished: %s", exp_dir)
