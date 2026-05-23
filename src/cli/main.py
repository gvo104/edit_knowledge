"""Главный entrypoint полного пайплайна эксперимента."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.experiments.experiment_manager import ExperimentManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Editing Full Pipeline")
    parser.add_argument(
        "--config",
        default=str(Path(config.BASE_DIR) / "configs" / "experiment" / "full_pipeline.yaml"),
        help="Путь к конфигурации полного эксперимента",
    )
    parser.add_argument(
        "--output_root",
        default=str(Path(config.BASE_DIR) / "outputs"),
        help="Корневая папка для результатов",
    )
    args = parser.parse_args()

    manager = ExperimentManager(output_root=Path(args.output_root))
    spec = manager.from_config(Path(args.config))
    result = manager.run(spec)

    print("=" * 70)
    print("FULL PIPELINE FINISHED")
    print(f"Report: {result['report_path']}")
    print(f"Aggregate: {result['aggregate_path']}")
    print(f"Stage runs: {result['num_stage_runs']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
