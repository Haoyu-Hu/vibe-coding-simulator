#!/usr/bin/env python3
"""Vibe-coding experiment runner — all three roles via API.

Developer, user simulator, and tester all use API-based inference.
The developer backend supports OpenRouter (default) or a direct
OpenAI-compatible endpoint via OPENAI_API_KEY / OPENAI_BASE_URL.

Usage:
    .venv/bin/python run_vibe_coding_api.py \\
        --datasets bigcodebench,humaneval \\
        --developer-model anthropic/claude-3.5-haiku \\
        --user-model openai/gpt-4o-mini \\
        --tester-model openai/gpt-4o-mini \\
        --max-rounds 5 --workers 4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from utils.logging_utils import setup_logging
from utils.multiturn_chain import (
    API_CLOSED_MODELS,
    create_experiment_dir,
    load_benchmark_samples,
    load_resume_snapshot,
    save_dataset_snapshot,
    utc_now,
    write_json,
)
from utils.openrouter import OpenRouterError, chat_text, validate_model_id
from utils.vibe_coding_chain import (
    VibeCodingExperimentConfig,
    build_vibe_chains,
    run_vibe_coding_experiment,
)
from utils.vibe_coding_tester import VibeCodingTesterConfig
from utils.vibe_coding_user_sim import (
    DEFAULT_LEVEL_WEIGHTS,
    VibeCodingUserConfig,
    load_rubric_defaults,
)
from utils.wandb_multiturn import WandbMultiTurnMonitor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    defaults = load_rubric_defaults()
    dist_defaults = defaults.get("level_distribution", {})
    user_defaults = defaults.get("user_simulator_defaults", {})
    tester_defaults = defaults.get("tester_defaults", {})

    parser = argparse.ArgumentParser(
        description="Run vibe-coding experiment with API inference for all three roles."
    )

    # Dataset
    parser.add_argument(
        "--manifest",
        default="datasets/multiturn_benchmark/manifest.json",
        help="Path to benchmark manifest (fetch_multiturn_benchmark_datasets.py output).",
    )
    parser.add_argument(
        "--datasets",
        default="bigcodebench,humaneval",
        help="Comma-separated coding dataset names, or 'all'.",
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=2)

    # Developer
    parser.add_argument(
        "--developer-model",
        default="anthropic/claude-3.5-haiku",
        help="Developer model (OpenRouter model ID).",
    )
    parser.add_argument(
        "--developer-provider",
        choices=["openrouter", "openai"],
        default="openrouter",
        help="Developer inference provider.",
    )
    parser.add_argument("--developer-temperature", type=float, default=0.2)
    parser.add_argument("--developer-max-tokens", type=int, default=1500)
    parser.add_argument("--developer-request-timeout", type=int, default=120)

    # User simulator
    parser.add_argument(
        "--user-model",
        default="openai/gpt-4o-mini",
        help="User simulator model (OpenRouter model ID).",
    )
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=float(user_defaults.get("temperature", 0.7)),
    )
    parser.add_argument(
        "--user-max-tokens",
        type=int,
        default=int(user_defaults.get("max_tokens", 300)),
    )
    parser.add_argument("--user-request-timeout", type=int, default=60)
    parser.add_argument(
        "--user-memory-window",
        type=int,
        default=int(user_defaults.get("memory_window", 2)),
        help="How many previous rounds the user simulator remembers.",
    )
    parser.add_argument(
        "--user-word-limit",
        type=int,
        default=int(user_defaults.get("word_limit", 120)),
    )

    # Tester
    parser.add_argument(
        "--tester-model",
        default="openai/gpt-4o-mini",
        help="Tester model (OpenRouter model ID).",
    )
    parser.add_argument("--tester-temperature", type=float, default=0.2)
    parser.add_argument("--tester-max-tokens", type=int, default=1200)
    parser.add_argument("--tester-request-timeout", type=int, default=90)
    parser.add_argument(
        "--public-test-budget",
        type=int,
        default=int(tester_defaults.get("public_test_budget", 4)),
        help="Max public tests per tester call.",
    )
    parser.add_argument(
        "--hidden-test-budget",
        type=int,
        default=int(tester_defaults.get("hidden_test_budget", 12)),
        help="Max hidden tests per tester call.",
    )
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=int(tester_defaults.get("test_timeout_seconds", 15)),
        help="Timeout for each test execution (seconds).",
    )

    # Experiment
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--consecutive-k", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Skip real API calls.")
    parser.add_argument("--skip-model-validation", action="store_true")

    # Knowledge level distribution
    parser.add_argument(
        "--level-weights",
        default=None,
        help=(
            "Comma-separated weight vector for knowledge levels 1-6 "
            "(e.g., '0.05,0.15,0.25,0.30,0.15,0.10'). "
            "Defaults to rubric defaults.json values."
        ),
    )

    # Provider routing (OpenRouter specific)
    parser.add_argument("--provider-order", default=None)
    parser.add_argument("--provider-only", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allow-fallbacks", action="store_true")
    group.add_argument("--no-fallbacks", action="store_true")
    parser.add_argument("--data-collection", choices=["allow", "deny"], default=None)

    # Output / resume
    parser.add_argument("--experiment-root", default="experiment-results")
    parser.add_argument("--label", default="vibe-coding-api")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-round", type=int, default=None)
    parser.add_argument("--log-dir", default="logs")

    # W&B
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.getenv("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-project", default="iterative-collapse-detection-prevention")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="vibe-coding,api")
    parser.add_argument("--wandb-strict", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _build_provider(args: argparse.Namespace) -> dict[str, Any] | None:
    provider: dict[str, Any] = {}
    if args.provider_order:
        provider["order"] = [x.strip() for x in args.provider_order.split(",") if x.strip()]
    if args.provider_only:
        provider["only"] = [x.strip() for x in args.provider_only.split(",") if x.strip()]
    if args.allow_fallbacks:
        provider["allow_fallbacks"] = True
    if args.no_fallbacks:
        provider["allow_fallbacks"] = False
    if args.data_collection:
        provider["data_collection"] = args.data_collection
    return provider or None


def _parse_level_weights(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        weights = [float(x.strip()) for x in raw.split(",")]
        if len(weights) != 6:
            raise ValueError("Need exactly 6 weights")
        return weights
    except Exception as exc:
        raise ValueError(f"Invalid --level-weights '{raw}': {exc}") from exc


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    logger, log_path = setup_logging("run_vibe_coding_api", Path(args.log_dir))

    level_weights = _parse_level_weights(args.level_weights)
    provider = _build_provider(args)

    # --- Resume or fresh start ---
    if args.resume_dir:
        exp_dir = Path(args.resume_dir).resolve()
        chains, restored_round = load_resume_snapshot(exp_dir, args.resume_round)
        start_round = restored_round + 1
        logger.info("Resuming vibe-coding API run from %s at round %s", exp_dir, restored_round)
    else:
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.exists():
            logger.error("Manifest not found: %s", manifest_path)
            logger.error("Run fetch_multiturn_benchmark_datasets.py first.")
            return 2

        selected_datasets = None if args.datasets == "all" else [
            x.strip() for x in args.datasets.split(",") if x.strip()
        ]
        try:
            samples = load_benchmark_samples(
                manifest_path=manifest_path,
                selected_datasets=selected_datasets,
                max_samples_per_dataset=args.max_samples_per_dataset,
            )
        except Exception as exc:
            logger.error("Failed to load benchmark samples: %s", exc)
            return 2
        if not samples:
            logger.error("No samples loaded.")
            return 2

        developer_profiles = [
            {"profile_name": f"api-{args.developer_model.replace('/', '-')}", "model": args.developer_model}
        ]
        chains = build_vibe_chains(
            samples=samples,
            developer_profiles=developer_profiles,
            level_weights=level_weights,
        )
        if not chains:
            logger.error("No vibe-coding chains built. Ensure datasets are coding tasks.")
            return 2

        exp_dir = create_experiment_dir(Path(args.experiment_root).resolve(), args.label)
        save_dataset_snapshot(samples, exp_dir)
        write_json(
            exp_dir / "config.json",
            {
                "runner": "vibe-coding-api",
                "created_at": utc_now(),
                "manifest": str(manifest_path),
                "datasets": selected_datasets or "all",
                "developer_model": args.developer_model,
                "developer_provider": args.developer_provider,
                "developer_temperature": args.developer_temperature,
                "developer_max_tokens": args.developer_max_tokens,
                "user_model": args.user_model,
                "user_temperature": args.user_temperature,
                "user_max_tokens": args.user_max_tokens,
                "user_memory_window": args.user_memory_window,
                "user_word_limit": args.user_word_limit,
                "tester_model": args.tester_model,
                "tester_temperature": args.tester_temperature,
                "tester_max_tokens": args.tester_max_tokens,
                "public_test_budget": args.public_test_budget,
                "hidden_test_budget": args.hidden_test_budget,
                "test_timeout": args.test_timeout,
                "max_rounds": args.max_rounds,
                "consecutive_k": args.consecutive_k,
                "workers": args.workers,
                "seed": args.seed,
                "dry_run": args.dry_run,
                "level_weights": level_weights or DEFAULT_LEVEL_WEIGHTS,
                "max_samples_per_dataset": args.max_samples_per_dataset,
                "provider": provider or {},
                "wandb_mode": args.wandb_mode,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_run_name": args.wandb_run_name,
                "wandb_tags": args.wandb_tags,
                "log_path": str(log_path),
                "num_chains": len(chains),
            },
        )
        start_round = 1

    # --- API key setup ---
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not args.dry_run:
        if not api_key and not openai_api_key:
            logger.error("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY for --developer-provider openai).")
            return 2

        if not args.skip_model_validation and api_key:
            for model_id in {args.developer_model, args.user_model, args.tester_model}:
                ok, suggestions = validate_model_id(api_key, model_id)
                if not ok:
                    logger.error(
                        "Unknown model '%s'. Suggestions: %s",
                        model_id,
                        ", ".join(suggestions) if suggestions else "none",
                    )
                    return 2

    # --- Inference functions ---
    def _openrouter_call(
        model: str, prompt: str, system_prompt: str | None,
        temperature: float, max_tokens: int, timeout: int,
    ) -> tuple[str, str]:
        try:
            text = chat_text(
                api_key=api_key,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                provider=provider,
            )
            return text, ""
        except OpenRouterError as exc:
            return "", str(exc)
        except Exception as exc:
            return "", f"{exc.__class__.__name__}: {exc}"

    def _openai_call(
        model: str, prompt: str, system_prompt: str | None,
        temperature: float, max_tokens: int, timeout: int,
    ) -> tuple[str, str]:
        """Direct OpenAI-compatible call (without OpenRouter)."""
        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = json.dumps({
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }).encode("utf-8")
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            }
            request = urllib.request.Request(
                url=f"{openai_base_url}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip(), ""
        except Exception as exc:
            return "", f"{exc.__class__.__name__}: {exc}"

    def developer_inference_fn(chain: dict, prompt: str, round_index: int) -> tuple[str, str]:
        if args.dry_run:
            return f"def solution(x):\n    return x  # dry-run round {round_index}", ""
        if args.developer_provider == "openai":
            return _openai_call(
                args.developer_model, prompt, None,
                args.developer_temperature, args.developer_max_tokens, args.developer_request_timeout,
            )
        return _openrouter_call(
            args.developer_model, prompt, None,
            args.developer_temperature, args.developer_max_tokens, args.developer_request_timeout,
        )

    def user_inference_fn(prompt: str, system_prompt: str) -> tuple[str, str]:
        if args.dry_run:
            return "Can you write the code for this task?", ""
        return _openai_call(
            args.user_model, prompt, system_prompt,
            args.user_temperature, args.user_max_tokens, args.user_request_timeout,
        )

    def tester_inference_fn(prompt: str, system_prompt: str) -> tuple[str, str]:
        if args.dry_run:
            # Return minimal valid JSON for dry-run
            dry_output = json.dumps({
                "reasoning": "dry-run",
                "public_tests": [{"description": "basic test", "code": "assert True"}],
                "hidden_tests": [{"description": "edge case", "code": "assert True"}],
                "public_feedback": "Dry-run tester: basic test passed.",
            })
            return dry_output, ""
        return _openai_call(
            args.tester_model, prompt, system_prompt,
            args.tester_temperature, args.tester_max_tokens, args.tester_request_timeout,
        )

    # --- Config objects ---
    user_config = VibeCodingUserConfig(
        model=args.user_model,
        temperature=args.user_temperature,
        max_tokens=args.user_max_tokens,
        request_timeout=args.user_request_timeout,
        word_limit=args.user_word_limit,
        memory_window=args.user_memory_window,
        level_weights=level_weights,
    )
    tester_config = VibeCodingTesterConfig(
        model=args.tester_model,
        temperature=args.tester_temperature,
        max_tokens=args.tester_max_tokens,
        request_timeout=args.tester_request_timeout,
        public_test_budget=args.public_test_budget,
        hidden_test_budget=args.hidden_test_budget,
        test_timeout=args.test_timeout,
    )
    exp_config = VibeCodingExperimentConfig(
        max_rounds=args.max_rounds,
        consecutive_k=args.consecutive_k,
        workers=max(1, args.workers),
        test_timeout=args.test_timeout,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # --- W&B setup ---
    tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
    wandb_config = {
        "runner": "vibe-coding-api",
        "exp_dir": str(exp_dir),
        "resume": bool(args.resume_dir),
        "developer_model": args.developer_model,
        "user_model": args.user_model,
        "tester_model": args.tester_model,
        "max_rounds": args.max_rounds,
        "consecutive_k": args.consecutive_k,
        "dry_run": args.dry_run,
        "num_chains": len(chains),
        "datasets": sorted({c["dataset_name"] for c in chains}),
    }
    wandb_monitor = None
    try:
        wandb_monitor = WandbMultiTurnMonitor.create(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name or exp_dir.name,
            tags=tags,
            config=wandb_config,
        )
    except Exception as exc:
        if args.wandb_strict:
            logger.error("Failed to initialize W&B: %s", exc)
            return 2
        logger.warning("W&B unavailable; continuing without logging: %s", exc)

    # --- Run experiment ---
    try:
        run_vibe_coding_experiment(
            chains=chains,
            exp_dir=exp_dir,
            developer_inference_fn=developer_inference_fn,
            user_inference_fn=user_inference_fn,
            tester_inference_fn=tester_inference_fn,
            user_config=user_config,
            tester_config=tester_config,
            exp_config=exp_config,
            logger=logger,
            start_round=start_round,
            round_hook=_make_vibe_wandb_hook(wandb_monitor) if wandb_monitor else None,
        )
    finally:
        if wandb_monitor is not None:
            try:
                wandb_monitor.finish(exp_dir)
            except Exception as exc:
                logger.warning("Failed to finalize W&B run: %s", exc)

    logger.info("Vibe-coding API experiment done: %s", exp_dir)
    print(exp_dir)
    return 0


# ---------------------------------------------------------------------------
# W&B hook for vibe-coding metrics
# ---------------------------------------------------------------------------

def _make_vibe_wandb_hook(monitor):
    """Return a round_hook compatible with WandbMultiTurnMonitor that logs vibe metrics."""
    def _hook(round_index: int, round_metrics: dict, condition_metrics: list) -> None:
        if monitor is None:
            return
        payload = {
            "round": round_index,
            "global/mean_hidden_pass_rate": round_metrics.get("mean_hidden_pass_rate"),
            "global/mean_public_pass_rate": round_metrics.get("mean_public_pass_rate"),
            "global/mean_code_similarity_prev": round_metrics.get("mean_code_similarity_prev"),
            "global/hidden_regression_count": round_metrics.get("total_hidden_regression_count"),
            "global/hidden_regression_accum": round_metrics.get("total_hidden_regression_accum"),
            "global/active_chains": round_metrics.get("active_chains"),
            "global/marked_out_chains": round_metrics.get("marked_out_chains"),
            "global/api_failure_count": round_metrics.get("api_failure_count"),
            "global/exec_failure_count": round_metrics.get("exec_failure_count"),
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        for row in condition_metrics:
            prefix = f"condition/{row['dataset_name']}/{row['developer_model']}/level{row['knowledge_level']}"
            payload.update({
                f"{prefix}/hidden_pass_rate": row.get("mean_hidden_pass_rate"),
                f"{prefix}/public_pass_rate": row.get("mean_public_pass_rate"),
            })
        try:
            monitor.run.log(payload, step=round_index)
        except Exception:
            pass
    return _hook


if __name__ == "__main__":
    raise SystemExit(main())
