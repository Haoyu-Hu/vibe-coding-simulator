from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "unknown"


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


class WandbMultiTurnMonitor:
    def __init__(self, run: Any, wandb_mod: Any):
        self.run = run
        self._wandb = wandb_mod
        self._history_rows: list[dict[str, Any]] = []
        self._finished = False
        self.run.define_metric("round")
        self.run.define_metric("global/*", step_metric="round")
        self.run.define_metric("condition/*", step_metric="round")

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        project: str,
        entity: str | None,
        run_name: str,
        tags: list[str],
        config: dict[str, Any],
    ) -> WandbMultiTurnMonitor | None:
        if mode == "disabled":
            return None
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - runtime env
            raise RuntimeError(
                "wandb is required for online/offline logging. "
                "Install dependencies in this topic venv and retry."
            ) from exc

        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name,
            mode=mode,
            config=config,
            tags=tags,
            job_type="multiturn-baseline",
        )
        return cls(run=run, wandb_mod=wandb)

    def log_round(
        self,
        round_index: int,
        round_metrics: dict[str, Any],
        condition_metrics: list[dict[str, Any]],
    ) -> None:
        global_payload = {
            "round": round_index,
            "global/round_accuracy": round_metrics.get("round_accuracy"),
            "global/mean_progress_score": round_metrics.get("mean_progress_score"),
            "global/mean_math_progress_score": round_metrics.get("mean_math_progress_score"),
            "global/mean_numerical_difference": round_metrics.get("mean_numerical_difference"),
            "global/mean_code_progress_score": round_metrics.get("mean_code_progress_score"),
            "global/mean_semantic_progress_score": round_metrics.get("mean_semantic_progress_score"),
            "global/mean_answer_similarity_prev": round_metrics.get("mean_answer_similarity_prev"),
            "global/mean_thinking_similarity_prev": round_metrics.get("mean_thinking_similarity_prev"),
            "global/mean_cot_monitor_score": round_metrics.get("mean_cot_monitor_score"),
            "global/mean_trace_score": round_metrics.get("mean_trace_score"),
            "global/swebench_resolution_rate": round_metrics.get("swebench_resolution_rate"),
            "global/changed_correct_to_wrong_this_round": round_metrics.get("changed_correct_to_wrong_this_round"),
            "global/changed_correct_to_wrong_accum": round_metrics.get("changed_correct_to_wrong_accum"),
            "global/active_chains": round_metrics.get("active_chains"),
            "global/marked_out_chains": round_metrics.get("marked_out_chains"),
            "global/num_chains": round_metrics.get("num_chains"),
        }
        payload = _clean_payload(global_payload)

        for row in condition_metrics:
            prefix = (
                f"condition/{_slug(str(row['condition']))}"
                f"/dataset/{_slug(str(row['dataset_name']))}"
                f"/model/{_slug(str(row['model_name']))}"
            )
            condition_payload = {
                f"{prefix}/accuracy": row.get("round_accuracy"),
                f"{prefix}/progress_score": row.get("mean_progress_score"),
                f"{prefix}/math_progress_score": row.get("mean_math_progress_score"),
                f"{prefix}/numerical_difference": row.get("mean_numerical_difference"),
                f"{prefix}/code_progress_score": row.get("mean_code_progress_score"),
                f"{prefix}/semantic_progress_score": row.get("mean_semantic_progress_score"),
                f"{prefix}/answer_similarity": row.get("mean_answer_similarity_prev"),
                f"{prefix}/thinking_similarity": row.get("mean_thinking_similarity_prev"),
                f"{prefix}/cot_score": row.get("mean_cot_monitor_score"),
                f"{prefix}/trace_score": row.get("mean_trace_score"),
                f"{prefix}/resolution_rate": row.get("swebench_resolution_rate"),
                f"{prefix}/changed_correct_to_wrong_this_round": row.get("changed_correct_to_wrong_this_round"),
                f"{prefix}/changed_correct_to_wrong_accum": row.get("changed_correct_to_wrong_accum"),
                f"{prefix}/marked_question_num": row.get("marked_out_chains"),
                f"{prefix}/num_questions": row.get("num_questions"),
            }
            payload.update(_clean_payload(condition_payload))

            self._history_rows.append(
                {
                    "round": round_index,
                    "dataset_name": row.get("dataset_name"),
                    "model_name": row.get("model_name"),
                    "condition": row.get("condition"),
                    "output_mode": row.get("output_mode"),
                    "context_mode": row.get("context_mode"),
                    "num_questions": row.get("num_questions"),
                    "round_accuracy": row.get("round_accuracy"),
                    "progress_score": row.get("mean_progress_score"),
                    "math_progress_score": row.get("mean_math_progress_score"),
                    "numerical_difference": row.get("mean_numerical_difference"),
                    "code_progress_score": row.get("mean_code_progress_score"),
                    "semantic_progress_score": row.get("mean_semantic_progress_score"),
                    "answer_similarity": row.get("mean_answer_similarity_prev"),
                    "thinking_similarity": row.get("mean_thinking_similarity_prev"),
                    "cot_score": row.get("mean_cot_monitor_score"),
                    "trace_score": row.get("mean_trace_score"),
                    "resolution_rate": row.get("swebench_resolution_rate"),
                    "changed_correct_to_wrong_this_round": row.get("changed_correct_to_wrong_this_round"),
                    "changed_correct_to_wrong_accum": row.get("changed_correct_to_wrong_accum"),
                    "marked_question_num": row.get("marked_out_chains"),
                    "active_chains": row.get("active_chains"),
                }
            )

        self.run.log(payload, step=round_index)

    def finish(self, exp_dir: Path) -> None:
        if self._finished:
            return
        self._finished = True

        if self._history_rows:
            columns = [
                "round",
                "dataset_name",
                "model_name",
                "condition",
                "output_mode",
                "context_mode",
                "num_questions",
                "round_accuracy",
                "progress_score",
                "math_progress_score",
                "numerical_difference",
                "code_progress_score",
                "semantic_progress_score",
                "answer_similarity",
                "thinking_similarity",
                "cot_score",
                "trace_score",
                "resolution_rate",
                "changed_correct_to_wrong_this_round",
                "changed_correct_to_wrong_accum",
                "marked_question_num",
                "active_chains",
            ]
            table = self._wandb.Table(columns=columns)
            for row in self._history_rows:
                table.add_data(*[row.get(col) for col in columns])
            self.run.log({"round_condition_metrics_table": table})

        artifact = self._wandb.Artifact(name=f"multiturn-results-{_slug(exp_dir.name)}", type="experiment-results")
        for rel_path in [
            "config.json",
            "state.json",
            "rounds/round_metrics.jsonl",
            "rounds/condition_metrics_round.jsonl",
            "summary/summary_dataframe.csv",
            "summary/summary_round_by_condition.csv",
        ]:
            path = exp_dir / rel_path
            if path.exists():
                artifact.add_file(str(path), name=rel_path)
        self.run.log_artifact(artifact)
        self.run.finish()
