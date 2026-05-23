from typing import Any, Dict

from src.editing.lora_sft import LoraSFTMethod


class LocFTBFAugMethod(LoraSFTMethod):
    """LocFT-BF + augmentation adapter (currently reuses LoRA SFT backend)."""

    name = "locft_bf_aug"

    def prepare(self, **kwargs) -> Dict[str, Any]:
        payload = super().prepare(**kwargs)
        payload["augmentation"] = True
        payload["note"] = "Augmentation hook placeholder: currently uses existing SFT pipeline"
        return payload
