import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import config
import yaml
from src.core.artifacts import ArtifactLayout
from src.core.bootstrap import bootstrap_registries
from src.core.run_context import RunContext
from src.core.registry import global_registries
from src.experiments.base import ExperimentInterface
from src.experiments.manifests import load_run_manifests, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExperimentPipeline(ExperimentInterface):
    """Stage-based orchestrator over legacy scripts with reproducible manifests."""

    def __init__(self, dataset_name: str, model_name: str, method_name: str, output_root: Optional[Path] = None):
        bootstrap_registries()
        if not global_registries.datasets.has(dataset_name):
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if not global_registries.models.has(model_name):
            raise ValueError(f"Unknown model: {model_name}")
        if not global_registries.methods.has(method_name):
            raise ValueError(f"Unknown method: {method_name}")

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.method_name = method_name
        self.output_root = output_root or (PROJECT_ROOT / "outputs")

    def _new_stage(self, stage: str, extra: Optional[Dict[str, Any]] = None):
        ctx = RunContext.create(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            method_name=self.method_name,
            stage=stage,
            extra=extra or {},
        )
        layout = ArtifactLayout(self.output_root, ctx.run_id)
        layout.ensure()
        return ctx, layout

    def _run_script(self, script_rel: str, args: list, log_path: Path) -> int:
        cmd = [sys.executable, str(PROJECT_ROOT / script_rel)] + args
        with open(log_path, "w", encoding="utf-8") as logf:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT)
        return result.returncode

    def _latest_file(self, pattern: str) -> Optional[str]:
        files = sorted(glob.glob(pattern), reverse=True)
        return files[0] if files else None

    def _find_latest_triplets(self) -> Optional[str]:
        pattern = str(Path(config.OUTPUT_DIR) / f"{self.dataset_name}_*_triplets.json")
        return self._latest_file(pattern)

    def _find_latest_baseline_probe(self) -> Optional[str]:
        model_short = self.model_name.split("/")[-1].replace("-Instruct", "")
        pattern = str(Path(config.PROBING_DIR) / f"{model_short}_{self.dataset_name}_*_probing_results.json")
        return self._latest_file(pattern)

    def _find_latest_lora_probe(self) -> Optional[str]:
        model_short = self.model_name.split("/")[-1].replace("-Instruct", "")
        pattern = str(Path(config.PROBING_DIR_LORA) / f"{model_short}_{self.dataset_name}_*_probing_results.json")
        return self._latest_file(pattern)

    def _save_run(self, ctx: RunContext, layout: ArtifactLayout, cmd: Optional[str] = None) -> None:
        config_payload = {
            "run_id": ctx.run_id,
            "timestamp": ctx.timestamp,
            "dataset": ctx.dataset_name,
            "model": ctx.model_name,
            "method": ctx.method_name,
            "stage": ctx.stage,
            "command": cmd or "",
            "extra": ctx.extra,
        }
        with open(layout.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_payload, f, sort_keys=False, allow_unicode=True)
        ctx.save_manifest(layout.manifest_path)

    def run_prepare_data(self, **kwargs) -> RunContext:
        ctx, layout = self._new_stage("prepare_data")
        cmd_args = ["--dataset", self.dataset_name]
        code = self._run_script("data_processing/run_processing.py", cmd_args, layout.logs_dir / "stage.log")
        ctx.extra["return_code"] = code
        triplets_file = self._find_latest_triplets()
        if triplets_file:
            ctx.output_artifacts["triplets"] = triplets_file
        self._save_run(ctx, layout, cmd=f"python data_processing/run_processing.py {' '.join(cmd_args)}")
        return ctx

    def run_baseline_probe(self, version: Optional[str] = None, **kwargs) -> RunContext:
        ctx, layout = self._new_stage("baseline_probe", extra={"version": version})
        args = ["--dataset", self.dataset_name]
        if version:
            args += ["--version", version]
        triplets_file = self._find_latest_triplets()
        if triplets_file:
            ctx.input_artifacts["triplets"] = triplets_file
        code = self._run_script("model_probing/run_probing.py", args, layout.logs_dir / "stage.log")
        ctx.extra["return_code"] = code
        probe_file = self._find_latest_baseline_probe()
        if probe_file:
            ctx.output_artifacts["baseline_probe"] = probe_file
            try:
                data = json.loads(Path(probe_file).read_text(encoding="utf-8"))
                write_json(layout.metrics_dir / "summary.json", data.get("summary", {}))
            except Exception:
                pass
        self._save_run(ctx, layout, cmd=f"python model_probing/run_probing.py {' '.join(args)}")
        return ctx

    def run_train_edit_method(self, version: Optional[str] = None, probe_after: bool = False, **kwargs) -> RunContext:
        ctx, layout = self._new_stage("train_edit_method", extra={"version": version})
        method_cls = global_registries.methods.get(self.method_name)
        method = method_cls()

        triplets_file = self._find_latest_triplets()
        if triplets_file:
            ctx.input_artifacts["triplets"] = triplets_file

        method.prepare(dataset_name=self.dataset_name, version=version)
        train_result = method.train_or_edit(
            dataset_name=self.dataset_name,
            version=version,
            probe_after=probe_after,
        )
        ctx.extra["train_result"] = train_result
        adapter_path = train_result.get("adapter_path", "")
        if adapter_path:
            ctx.output_artifacts["adapter"] = adapter_path
        self._save_run(ctx, layout, cmd=f"method={self.method_name} dataset={self.dataset_name} version={version}")
        return ctx

    def run_post_edit_probe(self, lora_path: str, version: Optional[str] = None, **kwargs) -> RunContext:
        ctx, layout = self._new_stage("post_edit_probe", extra={"version": version, "lora_path": lora_path})
        args = ["--dataset", self.dataset_name, "--lora_path", lora_path]
        if version:
            args += ["--version", version]
        ctx.input_artifacts["adapter"] = lora_path
        triplets_file = self._find_latest_triplets()
        if triplets_file:
            ctx.input_artifacts["triplets"] = triplets_file
        code = self._run_script("model_probing/run_probing.py", args, layout.logs_dir / "stage.log")
        ctx.extra["return_code"] = code
        probe_file = self._find_latest_lora_probe()
        if probe_file:
            ctx.output_artifacts["edited_probe"] = probe_file
            try:
                data = json.loads(Path(probe_file).read_text(encoding="utf-8"))
                write_json(layout.metrics_dir / "summary.json", data.get("summary", {}))
            except Exception:
                pass
        self._save_run(ctx, layout, cmd=f"python model_probing/run_probing.py {' '.join(args)}")
        return ctx

    def _build_comparison_report(self) -> Dict[str, Any]:
        manifests = load_run_manifests(self.output_root)
        baseline: Dict[str, Dict[str, Any]] = {}
        edited: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for m in manifests:
            dataset = m.get("dataset_name")
            method = m.get("method_name")
            stage = m.get("stage")
            out = m.get("output_artifacts", {})
            if stage == "baseline_probe" and out.get("baseline_probe"):
                try:
                    data = json.loads(Path(out["baseline_probe"]).read_text(encoding="utf-8"))
                    baseline[dataset] = data.get("summary", {})
                except Exception:
                    continue
            if stage == "post_edit_probe" and out.get("edited_probe"):
                try:
                    data = json.loads(Path(out["edited_probe"]).read_text(encoding="utf-8"))
                    edited.setdefault(dataset, {})[method] = data.get("summary", {})
                except Exception:
                    continue

        report: Dict[str, Any] = {"datasets": {}}
        for dataset, base_summary in baseline.items():
            report["datasets"][dataset] = {"baseline": base_summary, "methods": {}}
            for method_name, method_summary in edited.get(dataset, {}).items():
                categories = sorted(set(base_summary.keys()) | set(method_summary.keys()))
                deltas: Dict[str, Dict[str, float]] = {}
                for cat in categories:
                    b = base_summary.get(cat, {})
                    e = method_summary.get(cat, {})
                    deltas[cat] = {
                        "delta_mean_score": e.get("mean_score", 0.0) - b.get("mean_score", 0.0),
                        "delta_accuracy_1.0": e.get("accuracy_1.0", 0.0) - b.get("accuracy_1.0", 0.0),
                        "delta_good_enough_rate": e.get("good_enough_rate", 0.0) - b.get("good_enough_rate", 0.0),
                    }
                report["datasets"][dataset]["methods"][method_name] = {
                    "edited": method_summary,
                    "delta_vs_baseline": deltas,
                }
        return report

    def run_aggregate_results(self, **kwargs) -> RunContext:
        ctx, layout = self._new_stage("aggregate_results")
        report = self._build_comparison_report()
        write_json(layout.reports_dir / "comparison_report.json", report)
        ctx.output_artifacts["comparison_report"] = str(layout.reports_dir / "comparison_report.json")
        ctx.extra["datasets"] = sorted(report.get("datasets", {}).keys())
        self._save_run(ctx, layout, cmd="aggregate_from_manifests")
        return ctx
