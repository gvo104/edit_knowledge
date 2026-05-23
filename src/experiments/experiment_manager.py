from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.experiments.pipeline import ExperimentPipeline
from src.experiments.manifests import write_json


@dataclass
class ExperimentSpec:
    name: str
    model: str
    datasets: List[str]
    methods: List[str]
    steps: List[str]
    version: Optional[str] = None
    probe_after_train: bool = True


class ExperimentManager:
    """Runs multi-dataset/multi-method experiments from one config."""

    def __init__(self, output_root: Path):
        self.output_root = output_root

    @staticmethod
    def from_config(path: Path) -> ExperimentSpec:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)

        exp = payload.get("experiment", {})
        return ExperimentSpec(
            name=exp.get("name", "unnamed_experiment"),
            model=exp["model"],
            datasets=exp["datasets"],
            methods=exp["methods"],
            steps=exp.get(
                "steps",
                ["prepare_data", "baseline_probe", "train_edit_method", "post_edit_probe", "aggregate_results"],
            ),
            version=exp.get("version"),
            probe_after_train=bool(exp.get("probe_after_train", True)),
        )

    def run(self, spec: ExperimentSpec) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "experiment_name": spec.name,
            "model": spec.model,
            "datasets": spec.datasets,
            "methods": spec.methods,
            "runs": [],
        }

        for dataset_name in spec.datasets:
            baseline_pipeline = ExperimentPipeline(
                dataset_name=dataset_name,
                model_name=spec.model,
                method_name=spec.methods[0],
                output_root=self.output_root,
            )

            if "prepare_data" in spec.steps:
                ctx = baseline_pipeline.run_prepare_data()
                report["runs"].append(ctx.to_dict())

            if "baseline_probe" in spec.steps:
                ctx = baseline_pipeline.run_baseline_probe(version=spec.version)
                report["runs"].append(ctx.to_dict())

            for method_name in spec.methods:
                method_pipeline = ExperimentPipeline(
                    dataset_name=dataset_name,
                    model_name=spec.model,
                    method_name=method_name,
                    output_root=self.output_root,
                )

                train_ctx = None
                if "train_edit_method" in spec.steps:
                    train_ctx = method_pipeline.run_train_edit_method(
                        version=spec.version,
                        probe_after=False,
                    )
                    report["runs"].append(train_ctx.to_dict())

                if "post_edit_probe" in spec.steps:
                    lora_path = ""
                    if train_ctx:
                        lora_path = train_ctx.output_artifacts.get("adapter", "")
                    if lora_path:
                        probe_ctx = method_pipeline.run_post_edit_probe(
                            lora_path=lora_path,
                            version=spec.version,
                        )
                        report["runs"].append(probe_ctx.to_dict())

        if "aggregate_results" in spec.steps:
            aggregator_pipeline = ExperimentPipeline(
                dataset_name=spec.datasets[0],
                model_name=spec.model,
                method_name=spec.methods[0],
                output_root=self.output_root,
            )
            ctx = aggregator_pipeline.run_aggregate_results()
            report["runs"].append(ctx.to_dict())

        reports_dir = self.output_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_json(reports_dir / f"{spec.name}_{ts}_run_report.json", report)
        return report
