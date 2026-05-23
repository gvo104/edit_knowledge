"""Запуск полного эксперимента по одному конфигу."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

import config
from src.experiments.pipeline import ExperimentPipeline
from src.probing.metrics.schema import CategoryMetrics, ProbingMetrics


@dataclass
class ExperimentSpec:
    """Спецификация полного эксперимента."""

    name: str
    model: str
    datasets: List[str]
    methods: List[str]
    sample_size: int
    batch_size: int
    max_new_tokens: int
    evaluation_strategy: str
    verbose: bool


class ExperimentManager:
    """Оркестратор полного запуска: от данных до финального отчёта."""

    def __init__(self, output_root: Path):
        self.output_root = output_root

    @staticmethod
    def from_config(path: Path) -> ExperimentSpec:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        exp = payload.get("experiment", {})

        return ExperimentSpec(
            name=exp.get("name", "full_pipeline_v1"),
            model=exp.get("model", config.MODEL_NAME),
            datasets=exp.get("datasets", [config.DATASET_NAME]),
            methods=exp.get("methods", ["locft_bf", "locft_bf_aug", "rome"]),
            sample_size=int(exp.get("sample_size", config.SAMPLE_SIZE)),
            batch_size=int(exp.get("batch_size", config.BATCH_SIZE)),
            max_new_tokens=int(exp.get("max_new_tokens", config.MAX_NEW_TOKENS)),
            evaluation_strategy=str(exp.get("evaluation_strategy", config.EVALUATION_STRATEGY)),
            verbose=bool(exp.get("verbose", config.VERBOSE)),
        )

    @staticmethod
    def _load_metrics(path: str) -> ProbingMetrics:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = {}
        for k, v in data.get("categories", {}).items():
            categories[k] = CategoryMetrics(
                total=int(v.get("total", 0)),
                perfect_1_0=int(v.get("perfect_1_0", 0)),
                accuracy_1_0=float(v.get("accuracy_1_0", 0.0)),
                good_enough_count=int(v.get("good_enough_count", 0)),
                good_enough_rate=float(v.get("good_enough_rate", 0.0)),
                mean_score=float(v.get("mean_score", 0.0)),
            )
        return ProbingMetrics(categories=categories)

    def run(self, spec: ExperimentSpec) -> Dict[str, Any]:
        run_params = {
            "sample_size": spec.sample_size,
            "batch_size": spec.batch_size,
            "max_new_tokens": spec.max_new_tokens,
            "evaluation_strategy": spec.evaluation_strategy,
            "verbose": spec.verbose,
        }

        baseline_by_dataset: Dict[str, ProbingMetrics] = {}
        edited_by_dataset_method: Dict[str, Dict[str, ProbingMetrics]] = {}
        stage_runs: List[Dict[str, Any]] = []

        for dataset in spec.datasets:
            prepare_pipe = ExperimentPipeline(
                experiment_name=spec.name,
                dataset_name=dataset,
                model_name=spec.model,
                method_name=spec.methods[0],
                output_root=self.output_root,
                run_params=run_params,
            )
            prepare_ctx = prepare_pipe.run_prepare_data()
            stage_runs.append(prepare_ctx.to_dict())
            triplets_path = prepare_ctx.output_artifacts["triplets"]

            baseline_ctx = prepare_pipe.run_baseline_probe(triplets_path=triplets_path)
            stage_runs.append(baseline_ctx.to_dict())
            baseline_metrics_path = baseline_ctx.output_artifacts["baseline_metrics"]
            baseline_by_dataset[dataset] = self._load_metrics(baseline_metrics_path)

            edited_by_dataset_method[dataset] = {}
            for method in spec.methods:
                method_pipe = ExperimentPipeline(
                    experiment_name=spec.name,
                    dataset_name=dataset,
                    model_name=spec.model,
                    method_name=method,
                    output_root=self.output_root,
                    run_params=run_params,
                )
                train_ctx = method_pipe.run_train_edit_method(triplets_path=triplets_path)
                stage_runs.append(train_ctx.to_dict())

                adapter_path = train_ctx.output_artifacts.get("adapter")
                if not adapter_path:
                    edited_by_dataset_method[dataset][method] = baseline_by_dataset[dataset]
                    continue

                post_ctx = method_pipe.run_post_edit_probe(
                    triplets_path=triplets_path,
                    adapter_path=adapter_path,
                )
                stage_runs.append(post_ctx.to_dict())
                edited_metrics_path = post_ctx.output_artifacts["edited_metrics"]
                edited_by_dataset_method[dataset][method] = self._load_metrics(edited_metrics_path)

        agg_pipe = ExperimentPipeline(
            experiment_name=spec.name,
            dataset_name=spec.datasets[0],
            model_name=spec.model,
            method_name=spec.methods[0],
            output_root=self.output_root,
            run_params=run_params,
        )
        meta = {
            "name": spec.name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "datasets": spec.datasets,
            "models": [spec.model],
            "methods": spec.methods,
            "run_params": run_params,
        }
        agg_ctx = agg_pipe.run_aggregate_results(
            baseline_by_dataset=baseline_by_dataset,
            edited_by_dataset_method=edited_by_dataset_method,
            report_meta=meta,
        )
        stage_runs.append(agg_ctx.to_dict())

        final_report_path = self.output_root / "reports"
        final_report_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = final_report_path / f"{spec.name}_{ts}_run_report.json"

        with open(agg_ctx.output_artifacts["comparison_report"], "r", encoding="utf-8") as f:
            comparison_payload = json.load(f)

        final_payload = {
            "metadata": meta,
            "stages": stage_runs,
            "comparison": comparison_payload.get("comparison", {}),
            "artifacts": {
                "aggregate_report": agg_ctx.output_artifacts["comparison_report"],
            },
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)

        return {
            "report_path": str(out),
            "aggregate_path": agg_ctx.output_artifacts["comparison_report"],
            "num_stage_runs": len(stage_runs),
        }
