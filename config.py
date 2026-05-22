"""
Центральный файл конфигурации для проекта edit_knowledge.
Все настройки вынесены сюда.
Поддерживает выбор датасета и версионирование результатов.
"""
from pathlib import Path
from datetime import datetime

# Корень проекта – папка edit_knowledge/
BASE_DIR = Path(__file__).resolve().parent

# ========================
# ВЫБОР ДАТАСЕТА
# ========================
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
    }
}

def get_dataset_config(name=None):
    """Возвращает конфигурацию для указанного датасета (по умолчанию из DATASET_NAME)."""
    name = name or DATASET_NAME
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Неизвестный датасет: {name}. Доступные: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]

# ========================
# ОБРАБОТКА ДАННЫХ
# ========================
SAMPLE_SIZE = 50

# Максимальное количество генов, участвующих в парах ко-оккуренции
MAX_COOCCUR_GENES = 5

# ========================
# ВЕРСИОНИРОВАНИЕ
# ========================
VERSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ========================
# ВЫХОДНЫЕ ДАННЫЕ (триплеты)
# ========================
OUTPUT_DIR = BASE_DIR / "data" / "triplets"
OUTPUT_FILENAME_TEMPLATE = "{dataset}_{version}_triplets.json"

# ========================
# PROBING МОДЕЛИ
# ========================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = "float16"
MAX_NEW_TOKENS = 30
BATCH_SIZE = 16
EVALUATION_STRATEGY = "improved"

# Порог «хорошего» ответа для статистики (используется в улучшенной метрике)
GOOD_ENOUGH_THRESHOLD = 0.8

# ========================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ PROBING
# ========================
PROBING_DIR = BASE_DIR / "data" / "probing_results"
PROBING_DIR_LORA = BASE_DIR / "data" / "probing_results_lora"   # <-- новая папка
PROBING_OUTPUT_FILENAME_TEMPLATE = "{model_short}_{dataset}_{version}_probing_results.json"

# ========================
# ЛОГИРОВАНИЕ
# ========================
VERBOSE = True