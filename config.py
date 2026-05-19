"""
Центральный файл конфигурации для проекта edit_knowledge.
Все настройки вынесены сюда.
"""
from pathlib import Path

# Корень проекта – папка, в которой лежит этот config.py (edit_knowledge/)
BASE_DIR = Path(__file__).resolve().parent

# ========================
# ОБРАБОТКА ДАННЫХ (PubTator)
# ========================

# Путь к исходному файлу gene2pubtatorcentral (или .gz)
DATA_PATH = BASE_DIR / 'data' / 'PubTator' / 'gene2pubtatorcentral'

# Размер выборки уникальных PMID для построения триплетов
SAMPLE_SIZE = 50

# Максимальное количество генов, участвующих в парах ко-оккуренции
MAX_COOCCUR_GENES = 5

# ========================
# ВЫХОДНЫЕ ДАННЫЕ (триплеты)
# ========================

# Папка и имя файла для сохранения триплетов
OUTPUT_DIR = BASE_DIR / 'data' / 'triplets'
OUTPUT_FILENAME = 'gene2pubtator_triplets.json'

# ========================
# PROBING МОДЕЛИ
# ========================

# Входной файл с триплетами (результат предыдущего этапа)
TRIPLETS_PATH = BASE_DIR / 'data' / 'triplets' / 'gene2pubtator_triplets.json'

# Модель для тестирования
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Устройство и тип данных
DEVICE = "cuda"          # "cpu" или "cuda"
TORCH_DTYPE = "float16"  # "float16", "bfloat16", "float32"

# Параметры генерации (жадный режим, наиболее быстрый и детерминированный)
MAX_NEW_TOKENS = 30      # достаточно для PMID или короткого названия гена
BATCH_SIZE = 16          # размер батча – регулируйте под свою видеопамять

# Стратегия оценки ответов ("basic" или "improved")
EVALUATION_STRATEGY = "improved"

# Порог «хорошего» ответа для статистики (используется в улучшенной метрике)
GOOD_ENOUGH_THRESHOLD = 0.8

# ========================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ PROBING
# ========================

PROBING_DIR = BASE_DIR / 'data' / 'probing_results'
PROBING_OUTPUT_FILENAME = 'qwen2.5_0.5b_probing_results.json'

# ========================
# ЛОГИРОВАНИЕ
# ========================
VERBOSE = True