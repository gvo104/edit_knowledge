import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class RunContext:
    experiment_name: str
    run_id: str
    timestamp: str
    dataset_name: str
    model_name: str
    method_name: str
    stage: str
    input_artifacts: Dict[str, str] = field(default_factory=dict)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_name: str,
        dataset_name: str,
        model_name: str,
        method_name: str,
        stage: str,
        run_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "RunContext":
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rid = run_id or f"{ts}_{uuid4().hex[:8]}"
        return cls(
            experiment_name=experiment_name,
            run_id=rid,
            timestamp=ts,
            dataset_name=dataset_name,
            model_name=model_name,
            method_name=method_name,
            stage=stage,
            extra=extra or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
