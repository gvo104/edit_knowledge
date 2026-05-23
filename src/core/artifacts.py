from pathlib import Path


class ArtifactLayout:
    """Структура артефактов запуска в человекочитаемом формате."""

    def __init__(
        self,
        base_output_dir: Path,
        experiment_name: str,
        dataset_name: str,
        model_name: str,
        method_name: str,
        stage: str,
        run_id: str,
    ) -> None:
        model_dir = model_name.replace("/", "__")
        self.run_dir = (
            base_output_dir
            / "runs"
            / experiment_name
            / dataset_name
            / model_dir
            / method_name
            / stage
            / run_id
        )
        self.config_path = self.run_dir / "config.yaml"
        self.manifest_path = self.run_dir / "manifest.json"
        self.logs_dir = self.run_dir / "logs"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.metrics_dir = self.run_dir / "metrics"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.reports_dir = self.run_dir / "reports"

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
