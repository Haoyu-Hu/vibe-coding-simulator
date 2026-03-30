#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


DATASET_SPECS = [
    {
        "name": "math500-full",
        "domain": "math",
        "task_type": "math",
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_name": None,
        "split": "test",
        "adapter": "math500",
    },
    {
        "name": "bigcodebench-full",
        "domain": "code",
        "task_type": "code",
        "hf_path": "bigcode/bigcodebench",
        "hf_name": None,
        "split": "v0.1.4",
        "adapter": "bigcodebench",
    },
    {
        "name": "naturalplan-full",
        "domain": "common",
        "task_type": "text",
        "hf_path": "tuandunghcmut/natural-plan-benchmark",
        "hf_name": None,
        "split": "test",
        "adapter": "naturalplan",
        "fallback_sources": [
            {
                "hf_path": "tuandunghcmut/naturalplan-benchmark",
                "hf_name": None,
                "split": "test",
            }
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and normalize complete multi-turn benchmark datasets from public sources."
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/multiturn_benchmark_full",
        help="Directory for normalized complete datasets.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names from spec list, or 'all'.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=0,
        help="Optional cap per dataset (0 means all available samples).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before optional capping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional cache directory passed to datasets.load_dataset.",
    )
    parser.add_argument(
        "--hf-token",
        default=(
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HF_API_TOKEN")
        ),
        help=(
            "Optional Hugging Face token. Not required for public datasets, "
            "but needed for gated/private datasets or restricted environments."
        ),
    )
    return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_candidates(spec: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "hf_path": spec["hf_path"],
            "hf_name": spec.get("hf_name"),
            "split": spec["split"],
        }
    ]
    for fallback in spec.get("fallback_sources", []):
        candidates.append(
            {
                "hf_path": fallback["hf_path"],
                "hf_name": fallback.get("hf_name"),
                "split": fallback.get("split", spec["split"]),
            }
        )
    return candidates


def _load_hf_rows(
    spec: dict[str, Any],
    hf_cache_dir: str | None,
    hf_token: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "datasets package is required for full dataset fetch. "
            "Install with: ./.venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    failures: list[str] = []
    for source in _source_candidates(spec):
        kwargs: dict[str, Any] = {
            "path": source["hf_path"],
            "split": source["split"],
        }
        if source.get("hf_name"):
            kwargs["name"] = source["hf_name"]
        if hf_cache_dir:
            kwargs["cache_dir"] = hf_cache_dir
        if hf_token:
            kwargs["token"] = hf_token

        try:
            ds = load_dataset(**kwargs)  # type: ignore[arg-type]
            return [dict(row) for row in ds], source
        except Exception as exc:  # pragma: no cover - network and remote state dependent
            exc_summary = " ".join(str(exc).split())
            failures.append(
                f"path={source['hf_path']}, name={source.get('hf_name')}, "
                f"split={source['split']} -> {exc.__class__.__name__}: {exc_summary}"
            )

    token_hint = (
        " If this is a gated/private dataset, pass --hf-token or set HF_TOKEN."
        if not hf_token
        else ""
    )
    details = "\n".join([f"- {x}" for x in failures])
    raise RuntimeError(
        "Failed to download dataset from Hugging Face after trying all source candidates. "
        "Check internet/DNS access and dataset identifiers."
        f"{token_hint}\nAttempts:\n{details}"
    )


def _extract_entry_point_from_tests(tests: list[str]) -> str:
    for raw in tests:
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", raw)
        if match:
            return match.group(1)
    return "solution"


def _adapt_math(
    *,
    dataset_name: str,
    row: dict[str, Any],
    question: str,
    answer: str,
    canonical_solution: str | None,
    source: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": source}
    if canonical_solution:
        metadata["canonical_solution"] = canonical_solution
    return {
        "sample_id": f"{dataset_name}::{task_id}",
        "dataset_name": dataset_name,
        "domain": "math",
        "task_type": "math",
        "question": question,
        "reference_answer": answer,
        "metadata": metadata,
    }


def _adapt_text(
    *,
    dataset_name: str,
    row: dict[str, Any],
    question: str,
    answer: str,
    source: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"{dataset_name}::{task_id}",
        "dataset_name": dataset_name,
        "domain": "common",
        "task_type": "text",
        "question": question,
        "reference_answer": answer,
        "metadata": {"source": source},
    }


def _normalize_text_reference(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # NaturalPlan golden_plan values can arrive as a quoted JSON string.
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded.strip()
        except Exception:
            pass
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def _adapt_code(
    *,
    dataset_name: str,
    row: dict[str, Any],
    prompt: str,
    answer: str,
    test_code: str,
    entry_point: str,
    source: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"{dataset_name}::{task_id}",
        "dataset_name": dataset_name,
        "domain": "code",
        "task_type": "code",
        "question": prompt,
        "reference_answer": answer,
        "metadata": {
            "source": source,
            "prompt": prompt,
            "entry_point": entry_point,
            "test": test_code,
            "public_test": test_code,
        },
    }


def _adapt_swebench(
    *,
    dataset_name: str,
    question: str,
    patch: str,
    source: dict[str, Any],
    task_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "sample_id": f"{dataset_name}::{task_id}",
        "dataset_name": dataset_name,
        "domain": "code",
        "task_type": "swebench",
        "question": question,
        "reference_answer": patch,
        "metadata": {
            "source": source,
            **metadata,
        },
    }
    return payload


def transform_row(
    spec: dict[str, Any],
    row: dict[str, Any],
    idx: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    adapter = spec["adapter"]

    if adapter == "math500":
        task_id = str(row.get("unique_id", row.get("id", idx)))
        question = str(row.get("problem", row.get("question", "")))
        answer = str(row.get("answer", ""))
        canonical_solution = str(row.get("solution", "")).strip() or None
        return _adapt_math(
            dataset_name=spec["name"],
            row=row,
            question=question,
            answer=answer,
            canonical_solution=canonical_solution,
            source=source,
            task_id=task_id,
        )

    if adapter == "bigcodebench":
        task_id = str(row.get("task_id", idx))
        prompt = str(row.get("instruct_prompt", row.get("complete_prompt", row.get("prompt", ""))))
        answer = str(row.get("canonical_solution", ""))
        test_code = str(row.get("test", ""))
        entry_point = str(row.get("entry_point", "task_func"))
        return _adapt_code(
            dataset_name=spec["name"],
            row=row,
            prompt=prompt,
            answer=answer,
            test_code=test_code,
            entry_point=entry_point,
            source=source,
            task_id=task_id,
        )

    if adapter == "naturalplan":
        task_id = str(row.get("idx", row.get("id", idx)))
        question = str(row.get("prompt_0shot", row.get("prompt", row.get("question", ""))))
        answer = _normalize_text_reference(row.get("golden_plan", row.get("answer", "")))
        return _adapt_text(
            dataset_name=spec["name"],
            row=row,
            question=question,
            answer=answer,
            source=source,
            task_id=task_id,
        )

    raise ValueError(f"Unsupported adapter: {adapter}")


def normalize_dataset(
    spec: dict[str, Any],
    *,
    hf_cache_dir: str | None,
    hf_token: str | None,
    max_samples: int,
    shuffle: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, used_source = _load_hf_rows(spec, hf_cache_dir, hf_token)
    normalized = [transform_row(spec, row, idx, used_source) for idx, row in enumerate(rows)]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(normalized)
    if max_samples > 0:
        normalized = normalized[:max_samples]

    # Ensure metadata payloads stay JSON-serializable.
    for row in normalized:
        row["metadata"] = _to_jsonable(row.get("metadata", {}))
    return normalized, used_source


def main() -> int:
    args = parse_args()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_names = None
    if args.datasets != "all":
        selected_names = {x.strip() for x in args.datasets.split(",") if x.strip()}

    selected_specs = [spec for spec in DATASET_SPECS if selected_names is None or spec["name"] in selected_names]
    if not selected_specs:
        raise ValueError("No dataset selected. Check --datasets names.")

    manifest: dict[str, Any] = {
        "schema_version": "multiturn-benchmark-v2-full",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    for spec in selected_specs:
        print(f"[fetch] {spec['name']} <- {spec['hf_path']} ({spec['split']})", flush=True)
        try:
            rows, used_source = normalize_dataset(
                spec,
                hf_cache_dir=args.hf_cache_dir,
                hf_token=args.hf_token,
                max_samples=args.max_samples_per_dataset,
                shuffle=args.shuffle,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[error] {spec['name']}: {exc}", file=sys.stderr)
            return 1
        print(
            "[source] "
            f"path={used_source['hf_path']}, name={used_source.get('hf_name')}, "
            f"split={used_source['split']}",
            flush=True,
        )
        out_file = f"{spec['name']}.jsonl"
        _write_jsonl(out_dir / out_file, rows)
        manifest["datasets"].append(
            {
                "dataset_name": spec["name"],
                "file": out_file,
                "domain": spec["domain"],
                "task_type": spec["task_type"],
                "num_samples": len(rows),
                "source": {
                    "hf_path": used_source["hf_path"],
                    "hf_name": used_source.get("hf_name"),
                    "split": used_source["split"],
                    "adapter": spec["adapter"],
                },
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote full benchmark datasets to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
