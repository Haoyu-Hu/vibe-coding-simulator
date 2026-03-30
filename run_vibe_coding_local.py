#!/usr/bin/env python3
"""Vibe-coding experiment runner — local HF developer + API user/tester.

The developer uses a local Hugging Face model (or toy backend).
User simulator and tester are API-only (via OpenRouter).

Usage:
    # Toy mode (no API calls, no model download)
    .venv/bin/python run_vibe_coding_local.py \\
        --datasets bigcodebench --local-backend toy \\
        --user-backend toy --tester-backend toy --dry-run

    # Real HF developer, API user/tester
    .venv/bin/python run_vibe_coding_local.py \\
        --datasets bigcodebench \\
        --developer-model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --local-backend hf \\
        --user-model openai/gpt-4o-mini \\
        --tester-model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from utils.logging_utils import setup_logging
from utils.multiturn_chain import (
    LOCAL_MODEL_MAPS,
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
from run_vibe_coding_api import _make_vibe_wandb_hook, _parse_level_weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    defaults = load_rubric_defaults()
    user_defaults = defaults.get("user_simulator_defaults", {})
    tester_defaults = defaults.get("tester_defaults", {})

    parser = argparse.ArgumentParser(
        description="Run vibe-coding experiment with local developer + API user/tester."
    )

    # Dataset
    parser.add_argument(
        "--manifest",
        default="datasets/multiturn_benchmark/manifest.json",
        help="Path to benchmark manifest.",
    )
    parser.add_argument(
        "--datasets",
        default="bigcodebench,humaneval",
        help="Comma-separated coding dataset names, or 'all'.",
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=2)

    # Developer (local)
    parser.add_argument(
        "--developer-model",
        default=LOCAL_MODEL_MAPS["qwen"]["code"],
        help="HF model ID for the developer (or ignored in toy mode).",
    )
    parser.add_argument(
        "--local-backend",
        choices=["toy", "hf"],
        default="toy",
        help="toy: deterministic toy model. hf: real local HF model.",
    )
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--hf-max-new-tokens", type=int, default=1024)
    parser.add_argument("--hf-temperature", type=float, default=0.2)

    # User simulator
    parser.add_argument(
        "--user-backend",
        choices=["api", "toy"],
        default="api",
        help="api: OpenRouter model. toy: deterministic fallback.",
    )
    parser.add_argument("--user-model", default="openai/gpt-4o-mini")
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
    )
    parser.add_argument(
        "--user-word-limit",
        type=int,
        default=int(user_defaults.get("word_limit", 120)),
    )
    parser.add_argument("--skip-user-model-validation", action="store_true")

    # Tester
    parser.add_argument(
        "--tester-backend",
        choices=["api", "toy"],
        default="api",
        help="api: OpenRouter model. toy: deterministic pass-through.",
    )
    parser.add_argument("--tester-model", default="openai/gpt-4o-mini")
    parser.add_argument("--tester-temperature", type=float, default=0.2)
    parser.add_argument("--tester-max-tokens", type=int, default=1200)
    parser.add_argument("--tester-request-timeout", type=int, default=90)
    parser.add_argument(
        "--public-test-budget",
        type=int,
        default=int(tester_defaults.get("public_test_budget", 4)),
    )
    parser.add_argument(
        "--hidden-test-budget",
        type=int,
        default=int(tester_defaults.get("hidden_test_budget", 12)),
    )
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=int(tester_defaults.get("test_timeout_seconds", 15)),
    )

    # Experiment
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--consecutive-k", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")

    # Knowledge level distribution
    parser.add_argument(
        "--level-weights",
        default=None,
        help="Comma-separated weights for levels 1-6.",
    )

    # Output / resume
    parser.add_argument("--experiment-root", default="experiment-results")
    parser.add_argument("--label", default="vibe-coding-local")
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
    parser.add_argument("--wandb-tags", default="vibe-coding,local")
    parser.add_argument("--wandb-strict", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Toy developer (deterministic, for smoke tests)
# ---------------------------------------------------------------------------

def _toy_developer_response(chain: dict, round_index: int) -> str:
    task = str(chain.get("question", "")).strip()[:80]
    entry_point = str(chain.get("metadata", {}).get("entry_point", "solution"))
    ref = str(chain.get("reference_answer", "")).strip()
    if ref and "def " in ref:
        return ref[:600]
    return (
        f"def {entry_point}(*args, **kwargs):\n"
        f"    # TODO: implement solution for: {task[:50]}\n"
        f"    pass\n"
    )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    logger, log_path = setup_logging("run_vibe_coding_local", Path(args.log_dir))

    level_weights = _parse_level_weights(args.level_weights)
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    # --- Resume or fresh start ---
    if args.resume_dir:
        exp_dir = Path(args.resume_dir).resolve()
        chains, restored_round = load_resume_snapshot(exp_dir, args.resume_round)
        start_round = restored_round + 1
        logger.info("Resuming vibe-coding local run from %s at round %s", exp_dir, restored_round)
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
            {"profile_name": f"local-{args.local_backend}-{args.developer_model.replace('/', '-')[:30]}", "model": args.developer_model}
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
                "runner": "vibe-coding-local",
                "created_at": utc_now(),
                "manifest": str(manifest_path),
                "datasets": selected_datasets or "all",
                "developer_model": args.developer_model,
                "local_backend": args.local_backend,
                "hf_cache_dir": args.hf_cache_dir,
                "hf_device_map": args.hf_device_map,
                "hf_max_new_tokens": args.hf_max_new_tokens,
                "hf_temperature": args.hf_temperature,
                "user_backend": args.user_backend,
                "user_model": args.user_model,
                "user_temperature": args.user_temperature,
                "user_max_tokens": args.user_max_tokens,
                "user_memory_window": args.user_memory_window,
                "user_word_limit": args.user_word_limit,
                "tester_backend": args.tester_backend,
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

    # --- Validate API key for API roles ---
    if not args.dry_run:
        needs_api = (args.user_backend == "api") or (args.tester_backend == "api")
        if needs_api and not api_key:
            logger.error("Missing OPENROUTER_API_KEY for API-based user/tester roles.")
            return 2
        if api_key and not args.skip_user_model_validation:
            models_to_check = []
            if args.user_backend == "api":
                models_to_check.append(args.user_model)
            if args.tester_backend == "api":
                models_to_check.append(args.tester_model)
            for model_id in set(models_to_check):
                ok, suggestions = validate_model_id(api_key, model_id)
                if not ok:
                    logger.error(
                        "Unknown model '%s'. Suggestions: %s",
                        model_id, ", ".join(suggestions) if suggestions else "none",
                    )
                    return 2

    # --- Local HF generator setup ---
    hf_generator = None
    if args.local_backend == "hf":
        from utils.local_hf import HFBackendConfig, HFLocalGenerator
        hf_generator = HFLocalGenerator(
            HFBackendConfig(
                cache_dir=args.hf_cache_dir,
                device_map=args.hf_device_map,
                max_new_tokens=args.hf_max_new_tokens,
                temperature=args.hf_temperature,
            )
        )

    # --- Inference functions ---
    def developer_inference_fn(chain: dict, prompt: str, round_index: int) -> tuple[str, str]:
        if args.dry_run or args.local_backend == "toy":
            return _toy_developer_response(chain, round_index), ""
        assert hf_generator is not None
        try:
            output = hf_generator.generate(
                model_id=args.developer_model,
                prompt=prompt,
                system_prompt="",
            )
            return output, ""
        except Exception as exc:
            return "", str(exc)

    def _api_call(
        model: str, prompt: str, system_prompt: str,
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
                provider=None,
            )
            return text, ""
        except OpenRouterError as exc:
            return "", str(exc)
        except Exception as exc:
            return "", f"{exc.__class__.__name__}: {exc}"

    def user_inference_fn(prompt: str, system_prompt: str) -> tuple[str, str]:
        if args.dry_run or args.user_backend == "toy":
            return "", ""
        return _api_call(
            args.user_model, prompt, system_prompt,
            args.user_temperature, args.user_max_tokens, args.user_request_timeout,
        )

    def tester_inference_fn(prompt: str, system_prompt: str) -> tuple[str, str]:
        if args.dry_run or args.tester_backend == "toy":
            dry_output = json.dumps({
                "reasoning": "toy-mode",
                "public_tests": [{"description": "basic", "code": "assert True"}],
                "hidden_tests": [{"description": "edge", "code": "assert True"}],
                "public_feedback": "Toy tester: basic test passed.",
            })
            return dry_output, ""
        return _api_call(
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

    # --- W&B ---
    tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
    wandb_config = {
        "runner": "vibe-coding-local",
        "exp_dir": str(exp_dir),
        "resume": bool(args.resume_dir),
        "developer_model": args.developer_model,
        "local_backend": args.local_backend,
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

    logger.info("Vibe-coding local experiment done: %s", exp_dir)
    print(exp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
