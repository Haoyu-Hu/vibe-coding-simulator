#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from utils.multiturn_chain import LOCAL_MODEL_MAPS
from utils.local_hf import collect_local_model_ids, default_cache_dir, download_local_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local open-source models for offline multi-turn experiments."
    )
    parser.add_argument(
        "--family",
        choices=["qwen", "llama", "all"],
        default="all",
        help="Model family to download.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map passed to model loading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    cache_dir = args.cache_dir or default_cache_dir(root)

    model_ids = collect_local_model_ids(args.family, LOCAL_MODEL_MAPS)
    print("Downloading models:")
    for model_id in model_ids:
        print(f"- {model_id}")

    download_local_models(model_ids, cache_dir=cache_dir, device_map=args.device_map)
    print(f"Done. Models cached in: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
