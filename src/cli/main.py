import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.experiments.experiment_manager import ExperimentManager
from src.experiments.pipeline import ExperimentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified pipeline CLI")
    parser.add_argument("--step", required=False, choices=[
        "prepare_data",
        "baseline_probe",
        "train_edit_method",
        "post_edit_probe",
        "aggregate_results",
    ])
    parser.add_argument("--config", default=None, help="Path to full experiment yaml config")
    parser.add_argument("--run_full", action="store_true", help="Run full multi-dataset/multi-method experiment from --config")
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--method", default="lora_sft")
    parser.add_argument("--version", default=None)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--probe_after", action="store_true")
    parser.add_argument("--output_root", default=str(Path(config.BASE_DIR) / "outputs"))
    args = parser.parse_args()

    if args.run_full:
        if not args.config:
            raise ValueError("--config is required with --run_full")
        manager = ExperimentManager(output_root=Path(args.output_root))
        spec = manager.from_config(Path(args.config))
        report = manager.run(spec)
        print(f"Experiment '{spec.name}' finished. Stage runs: {len(report.get('runs', []))}")
        return

    if not args.step:
        raise ValueError("--step is required unless --run_full is used")

    pipeline = ExperimentPipeline(
        dataset_name=args.dataset,
        model_name=args.model,
        method_name=args.method,
        output_root=Path(args.output_root),
    )

    if args.step == "prepare_data":
        pipeline.run_prepare_data()
    elif args.step == "baseline_probe":
        pipeline.run_baseline_probe(version=args.version)
    elif args.step == "train_edit_method":
        pipeline.run_train_edit_method(version=args.version, probe_after=args.probe_after)
    elif args.step == "post_edit_probe":
        if not args.lora_path:
            raise ValueError("--lora_path is required for post_edit_probe")
        pipeline.run_post_edit_probe(lora_path=args.lora_path, version=args.version)
    elif args.step == "aggregate_results":
        pipeline.run_aggregate_results()


if __name__ == "__main__":
    main()
