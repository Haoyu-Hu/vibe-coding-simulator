#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "datasets"

DATASET_SPECS = [
    {"name": "math500-mini", "file": "math500_mini.jsonl", "domain": "math", "task_type": "math"},
    {"name": "bigcodebench-mini", "file": "bigcodebench_mini.jsonl", "domain": "code", "task_type": "code"},
    {"name": "naturalplan-mini", "file": "naturalplan_mini.jsonl", "domain": "common", "task_type": "text"},
    {"name": "humaneval", "file": "humaneval.jsonl", "domain": "code", "task_type": "code"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized multi-turn benchmark datasets from existing testing datasets."
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/multiturn_benchmark",
        help="Directory to store normalized benchmark datasets.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=0,
        help="Optional cap per dataset (0 means all).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def transform_row(spec: dict[str, str], row: dict[str, Any]) -> dict[str, Any]:
    dataset_name = spec["name"]
    task_id = row.get("task_id", "unknown")

    if spec["task_type"] == "math":
        question = row["question"]
        answer = row["answer"]
        metadata: dict[str, Any] = {}
        canonical_solution = row.get("solution")
        if canonical_solution:
            metadata["canonical_solution"] = canonical_solution
    elif spec["task_type"] == "code":
        question = row["prompt"]
        answer = row.get("reference_solution") or row.get("canonical_solution", "")
        metadata = {
            "prompt": row.get("prompt", ""),
            "entry_point": row.get("entry_point", "solution"),
            "test": row.get("test", ""),
            "public_test": row.get("public_test", ""),
        }
    else:
        question = row["question"]
        answer = row["answer"]
        metadata = {}

    return {
        "sample_id": f"{dataset_name}::{task_id}",
        "dataset_name": dataset_name,
        "domain": spec["domain"],
        "task_type": spec["task_type"],
        "question": question,
        "reference_answer": answer,
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "multiturn-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    for spec in DATASET_SPECS:
        source_path = SOURCE_DIR / spec["file"]
        if not source_path.exists():
            print(f"Skipping {spec['name']}: {source_path} not found")
            continue
        rows = read_jsonl(source_path)
        transformed = [transform_row(spec, row) for row in rows]
        if args.max_samples_per_dataset > 0:
            transformed = transformed[: args.max_samples_per_dataset]

        out_file = f"{spec['name']}.jsonl"
        write_jsonl(out_dir / out_file, transformed)
        manifest["datasets"].append(
            {
                "dataset_name": spec["name"],
                "file": out_file,
                "domain": spec["domain"],
                "task_type": spec["task_type"],
                "num_samples": len(transformed),
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote benchmark datasets to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
