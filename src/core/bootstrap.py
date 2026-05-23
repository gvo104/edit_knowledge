from typing import Dict

import config
from src.core.registry import global_registries
from src.data.pubtator_dataset import PubTatorDataset
from src.editing.locft_bf_aug import LocFTBFAugMethod
from src.editing.lora_sft import LoraSFTMethod
from src.editing.rome import RomeMethod
from src.editing.memit import MemitMethod
from src.editing.mend import MendMethod
from src.editing.wise import WiseMethod
from src.editing.model_merging import ModelMergingMethod


def bootstrap_registries() -> None:
    for ds_name, ds_cfg in config.DATASET_CONFIGS.items():
        if not global_registries.datasets.has(ds_name):
            global_registries.datasets.register(
                ds_name,
                {
                    "name": ds_name,
                    "config": ds_cfg,
                    "adapter": PubTatorDataset,
                },
            )

    if not global_registries.models.has(config.MODEL_NAME):
        global_registries.models.register(
            config.MODEL_NAME,
            {
                "name": config.MODEL_NAME,
                "device": config.DEVICE,
                "torch_dtype": config.TORCH_DTYPE,
            },
        )

    method_map = {
        "lora_sft": LoraSFTMethod,
        "locft_bf_aug": LocFTBFAugMethod,
        "rome": RomeMethod,
        "memit": MemitMethod,
        "mend": MendMethod,
        "wise": WiseMethod,
        "model_merging": ModelMergingMethod,
    }
    for method_name, method_cls in method_map.items():
        if not global_registries.methods.has(method_name):
            global_registries.methods.register(method_name, method_cls)

    if not global_registries.metrics.has("improved"):
        global_registries.metrics.register("improved", {"name": config.EVALUATION_STRATEGY})

    if not global_registries.run_configs.has("default"):
        global_registries.run_configs.register("default", get_default_run_config())


def get_default_run_config() -> Dict[str, str]:
    return {
        "dataset": config.DATASET_NAME,
        "model": config.MODEL_NAME,
        "method": "lora_sft",
        "metric": config.EVALUATION_STRATEGY,
    }
