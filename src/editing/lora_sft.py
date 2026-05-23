from pathlib import Path
from typing import Any, Dict, Optional

from src.editing.base import EditingMethod


class LoraSFTMethod(EditingMethod):
    """Adapter over existing LocFT-BF LoRA training implementation."""

    name = "lora_sft"

    def __init__(self) -> None:
        self._adapter_path: Optional[Path] = None

    def prepare(self, **kwargs) -> Dict[str, Any]:
        return {"status": "prepared", "params": kwargs}

    def train_or_edit(self, **kwargs) -> Dict[str, Any]:
        from model_training.run_sft_lora import train_lora

        dataset_name = kwargs["dataset_name"]
        version = kwargs.get("version")
        probe_after = kwargs.get("probe_after", False)
        force_retrain = kwargs.get("force_retrain", False)

        adapter_dir = train_lora(
            dataset_name=dataset_name,
            version=version,
            probe_after=probe_after,
            force_retrain=force_retrain,
        )
        self._adapter_path = Path(adapter_dir)
        return {"status": "trained", "adapter_path": str(self._adapter_path)}

    def save(self, **kwargs) -> Dict[str, Any]:
        if self._adapter_path is None:
            return {"status": "no_adapter"}
        return {"status": "saved", "adapter_path": str(self._adapter_path)}

    def load(self, **kwargs) -> Dict[str, Any]:
        adapter_path = kwargs.get("adapter_path")
        if not adapter_path:
            raise ValueError("adapter_path is required")
        self._adapter_path = Path(adapter_path)
        return {"status": "loaded", "adapter_path": str(self._adapter_path)}

    def apply_to_model(self, **kwargs) -> Dict[str, Any]:
        if self._adapter_path is None:
            raise ValueError("Adapter is not loaded or trained")
        return {"status": "applied", "adapter_path": str(self._adapter_path)}
