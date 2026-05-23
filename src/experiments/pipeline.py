"""Оркестрация стадий эксперимента без legacy-скриптов."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import config
import yaml
from src.core.artifacts import ArtifactLayout
from src.core.bootstrap import bootstrap_registries
from src.core.registry import global_registries
from src.core.run_context import RunContext
from src.experiments.base import ExperimentInterface
from src.probing.metrics.aggregator import build_comparison_report
from src.probing.metrics.calculator import compute_metrics_from_results
from src.probing.metrics.schema import ProbingMetrics


class ExperimentPipeline(ExperimentInterface):
    """Выполняет стадии пайплайна и сохраняет результаты в новой структуре."""

    def __init__(
        self,
        experiment_name: str,
        dataset_name: str,
        model_name: str,
        method_name: str,
        output_root: Path,
        run_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        bootstrap_registries()
        if not global_registries.datasets.has(dataset_name):
            raise ValueError(f"Неизвестный датасет: {dataset_name}")
        if not global_registries.models.has(model_name):
            raise ValueError(f"Неизвестная модель: {model_name}")
        if not global_registries.methods.has(method_name):
            raise ValueError(f"Неизвестный метод: {method_name}")

        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.method_name = method_name
        self.output_root = output_root
        self.run_params = run_params or {}

    def _new_stage(self, stage: str, extra: Optional[Dict[str, Any]] = None):
        ctx = RunContext.create(
            experiment_name=self.experiment_name,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            method_name=self.method_name,
            stage=stage,
            extra=extra or {},
        )
        layout = ArtifactLayout(
            base_output_dir=self.output_root,
            experiment_name=self.experiment_name,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            method_name=self.method_name,
            stage=stage,
            run_id=ctx.run_id,
        )
        layout.ensure()
        return ctx, layout

    def _save_run(self, ctx: RunContext, layout: ArtifactLayout, stage_cfg: Dict[str, Any]) -> None:
        with open(layout.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(stage_cfg, f, sort_keys=False, allow_unicode=True)
        ctx.save_manifest(layout.manifest_path)

    def run_prepare_data(self, **kwargs) -> RunContext:
        """Готовит триплеты и тестовые запросы из сырого датасета."""
        ctx, layout = self._new_stage("prepare_data")

        ds_meta = global_registries.datasets.get(self.dataset_name)
        ds_adapter_cls = ds_meta["adapter"]
        dataset = ds_adapter_cls(self.dataset_name)

        built = dataset.build_triplets(sample_size=self.run_params.get("sample_size", config.SAMPLE_SIZE))
        queries = dataset.build_queries(triplets=set(tuple(x) for x in built["triplets"]))

        payload = {
            "dataset": self.dataset_name,
            "entity_type": ds_meta["config"]["entity_type"],
            "triplets": built["triplets"],
            "test_queries": {k: v for k, v in queries.items() if k != "locality"},
            "locality_queries": queries.get("locality", []),
            "sample_info": {
                "total_pmids": len(built["entities_by_pmid"]),
                "total_annotations": len(built["sample_records"]),
                "total_unique_triplets": len(built["triplets"]),
            },
        }

        triplets_path = layout.artifacts_dir / "triplets.json"
        with open(triplets_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        ctx.output_artifacts["triplets"] = str(triplets_path)
        self._save_run(
            ctx,
            layout,
            stage_cfg={
                "stage": "prepare_data",
                "experiment": self.experiment_name,
                "dataset": self.dataset_name,
                "model": self.model_name,
                "method": self.method_name,
            },
        )
        return ctx

    def run_baseline_probe(self, version: Optional[str] = None, **kwargs) -> RunContext:
        """Запускает baseline probing и сохраняет нормализованные метрики."""
        triplets_path = kwargs.get("triplets_path")
        if not triplets_path:
            raise ValueError("triplets_path обязателен для baseline probing")

        ctx, layout = self._new_stage("baseline_probe")
        ctx.input_artifacts["triplets"] = str(triplets_path)

        with open(triplets_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_queries = data["test_queries"]
        test_queries["locality"] = data.get("locality_queries", [])

        from llm.model_loader import load_model_and_tokenizer
        from model_probing.probing_runner import run_probing

        tokenizer, model = load_model_and_tokenizer(
            self.model_name,
            torch_dtype=config.TORCH_DTYPE,
            device_map=config.DEVICE,
        )

        results = run_probing(
            test_queries,
            tokenizer,
            model,
            {
                "max_new_tokens": self.run_params.get("max_new_tokens", config.MAX_NEW_TOKENS),
                "batch_size": self.run_params.get("batch_size", config.BATCH_SIZE),
            },
            eval_strategy=self.run_params.get("evaluation_strategy", config.EVALUATION_STRATEGY),
            verbose=self.run_params.get("verbose", config.VERBOSE),
        )

        metrics = compute_metrics_from_results(results, good_enough_threshold=config.GOOD_ENOUGH_THRESHOLD)

        raw_path = layout.artifacts_dir / "baseline_probing_raw.json"
        metrics_path = layout.metrics_dir / "metrics.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        ctx.output_artifacts["baseline_raw"] = str(raw_path)
        ctx.output_artifacts["baseline_metrics"] = str(metrics_path)

        self._save_run(
            ctx,
            layout,
            stage_cfg={
                "stage": "baseline_probe",
                "experiment": self.experiment_name,
                "dataset": self.dataset_name,
                "model": self.model_name,
                "method": self.method_name,
            },
        )
        return ctx

    def run_train_edit_method(self, version: Optional[str] = None, **kwargs) -> RunContext:
        """Обучает/применяет метод редактирования и сохраняет артефакты."""
        triplets_path = kwargs.get("triplets_path")
        if not triplets_path:
            raise ValueError("triplets_path обязателен для train_edit_method")

        ctx, layout = self._new_stage("train_edit_method")
        ctx.input_artifacts["triplets"] = str(triplets_path)

        method_cls = global_registries.methods.get(self.method_name)
        method = method_cls()
        method.prepare(dataset_name=self.dataset_name, triplets_path=str(triplets_path), output_dir=str(layout.artifacts_dir))
        result = method.train_or_edit(
            dataset_name=self.dataset_name,
            triplets_path=str(triplets_path),
            output_dir=str(layout.artifacts_dir),
            model_name=self.model_name,
        )

        ctx.extra["train_result"] = result
        adapter_path = result.get("adapter_path", "")
        if adapter_path:
            ctx.output_artifacts["adapter"] = adapter_path

        self._save_run(
            ctx,
            layout,
            stage_cfg={
                "stage": "train_edit_method",
                "experiment": self.experiment_name,
                "dataset": self.dataset_name,
                "model": self.model_name,
                "method": self.method_name,
            },
        )
        return ctx

    def run_post_edit_probe(self, version: Optional[str] = None, **kwargs) -> RunContext:
        """Проверяет метрики после редактирования модели."""
        triplets_path = kwargs.get("triplets_path")
        adapter_path = kwargs.get("adapter_path")
        if not triplets_path or not adapter_path:
            raise ValueError("triplets_path и adapter_path обязательны для post_edit_probe")

        ctx, layout = self._new_stage("post_edit_probe")
        ctx.input_artifacts["triplets"] = str(triplets_path)
        ctx.input_artifacts["adapter"] = str(adapter_path)

        with open(triplets_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_queries = data["test_queries"]
        test_queries["locality"] = data.get("locality_queries", [])

        from llm.model_loader import load_model_and_tokenizer
        from model_probing.probing_runner import run_probing
        from peft import PeftModel

        tokenizer, base_model = load_model_and_tokenizer(
            self.model_name,
            torch_dtype=config.TORCH_DTYPE,
            device_map=config.DEVICE,
        )
        edited_model = PeftModel.from_pretrained(base_model, adapter_path)

        results = run_probing(
            test_queries,
            tokenizer,
            edited_model,
            {
                "max_new_tokens": self.run_params.get("max_new_tokens", config.MAX_NEW_TOKENS),
                "batch_size": self.run_params.get("batch_size", config.BATCH_SIZE),
            },
            eval_strategy=self.run_params.get("evaluation_strategy", config.EVALUATION_STRATEGY),
            verbose=self.run_params.get("verbose", config.VERBOSE),
        )

        metrics = compute_metrics_from_results(results, good_enough_threshold=config.GOOD_ENOUGH_THRESHOLD)

        raw_path = layout.artifacts_dir / "post_edit_probing_raw.json"
        metrics_path = layout.metrics_dir / "metrics.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        ctx.output_artifacts["edited_raw"] = str(raw_path)
        ctx.output_artifacts["edited_metrics"] = str(metrics_path)

        self._save_run(
            ctx,
            layout,
            stage_cfg={
                "stage": "post_edit_probe",
                "experiment": self.experiment_name,
                "dataset": self.dataset_name,
                "model": self.model_name,
                "method": self.method_name,
            },
        )
        return ctx

    def run_aggregate_results(self, **kwargs) -> RunContext:
        """Агрегирует baseline и edited метрики в финальный отчёт."""
        baseline_by_dataset: Dict[str, ProbingMetrics] = kwargs["baseline_by_dataset"]
        edited_by_dataset_method: Dict[str, Dict[str, ProbingMetrics]] = kwargs["edited_by_dataset_method"]
        report_meta: Dict[str, Any] = kwargs["report_meta"]

        ctx, layout = self._new_stage("aggregate_results")

        comparison = build_comparison_report(baseline_by_dataset, edited_by_dataset_method)

        final_report = {
            "experiment": report_meta,
            "comparison": comparison.to_dict(),
        }

        report_path = layout.reports_dir / "comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        ctx.output_artifacts["comparison_report"] = str(report_path)
        self._save_run(
            ctx,
            layout,
            stage_cfg={
                "stage": "aggregate_results",
                "experiment": self.experiment_name,
                "dataset": self.dataset_name,
                "model": self.model_name,
                "method": self.method_name,
            },
        )
        return ctx
