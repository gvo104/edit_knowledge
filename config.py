"""Базовые настройки проекта для нового модульного пайплайна."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_NAME = "gene"
DATASET_CONFIGS = {
    "gene": {
        "path": BASE_DIR / "data" / "PubTator" / "gene2pubtatorcentral",
        "id_field": "gene_id",
        "name_field": "gene_name",
        "entity_type": "gene",
    },
    "disease": {
        "path": BASE_DIR / "data" / "PubTator" / "disease2pubtatorcentral",
        "id_field": "disease_id",
        "name_field": "disease_name",
        "entity_type": "disease",
    },
    "mutation": {
        "path": BASE_DIR / "data" / "PubTator" / "mutation2pubtatorcentral",
        "id_field": "mutation_id",
        "name_field": "mutation_name",
        "entity_type": "mutation",
    },
}


def get_dataset_config(name=None):
    """Возвращает конфигурацию датасета по имени."""
    selected = name or DATASET_NAME
    if selected not in DATASET_CONFIGS:
        raise ValueError(f"Неизвестный датасет: {selected}")
    return DATASET_CONFIGS[selected]


SAMPLE_SIZE = 50
MAX_COOCCUR_GENES = 5

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = "float16"
MAX_NEW_TOKENS = 30
BATCH_SIZE = 16
EVALUATION_STRATEGY = "improved"
GOOD_ENOUGH_THRESHOLD = 0.8

AUGMENT_PARAPHRASES = False
VERBOSE = True
