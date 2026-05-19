"""
Основной скрипт обработки данных.
Запускает всю цепочку: статистика → выборка → триплеты → тестовые запросы → сохранение.
"""
import sys
import json
import os
from pathlib import Path

# === Настройка путей для импорта общего конфига ===
# Добавляем корень проекта (на одну папку выше, чем этот скрипт) в sys.path,
# чтобы можно было сделать import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # Теперь импортируется из корня проекта

# Локальные модули обработки данных (лежат в этой же папке)
from data_reader import iter_records
from stats import compute_statistics
from triplet_extractor import extract_sample, group_genes_by_pmid, build_triplets
from query_generator import generate_queries


def main():
    # 1. Чтение конфигурации
    data_path = Path(config.DATA_PATH)
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === ВЫВОД ЗАГОЛОВКА ===
    print("=" * 60)
    print("ОБРАБОТКА ДАННЫХ: извлечение SPO-триплетов из gene2pubtatorcentral")
    print("=" * 60)

    # 2. Первый проход – статистика (потоковый)
    print("\n>>> Этап 1: Сбор статистики по всему файлу...")
    stats = compute_statistics(iter_records(data_path), verbose=config.VERBOSE)
    print(f"    Всего записей: {stats['total_lines']}")
    print(f"    Уникальных PMID: {stats['unique_pmids']}")
    print(f"    Уникальных генов: {stats['unique_genes']}")

    # 3. Второй проход – выборка первых SAMPLE_SIZE уникальных PMID
    print(f"\n>>> Этап 2: Загрузка выборки из первых {config.SAMPLE_SIZE} статей...")
    sample_records = extract_sample(iter_records(data_path), config.SAMPLE_SIZE)
    print(f"    Записей в выборке: {len(sample_records)}")

    # 4. Группировка генов по PMID
    print("\n>>> Этап 3: Группировка генов по статьям...")
    genes_by_pmid = group_genes_by_pmid(sample_records)
    print(f"    Статей в выборке: {len(genes_by_pmid)}")

    # 5. Построение триплетов
    print("\n>>> Этап 4: Создание SPO-триплетов...")
    triplets = build_triplets(genes_by_pmid, config.MAX_COOCCUR_GENES)
    unique_triplets = list(triplets)  # для сериализации превращаем в список
    print(f"    Уникальных триплетов: {len(unique_triplets)}")

    # 6. Генерация тестовых запросов
    print("\n>>> Этап 5: Генерация тестовых запросов...")
    test_queries = generate_queries(triplets)
    for key in test_queries:
        print(f"    {key}: {len(test_queries[key])} шт.")

    # 7. Сохранение результата
    output_path = output_dir / config.OUTPUT_FILENAME
    output = {
        "source": "gene2pubtatorcentral",
        "sample_info": {
            "total_pmids": len(genes_by_pmid),
            "total_annotations": len(sample_records),
            "total_unique_triplets": len(unique_triplets),
            "sample_size_limit": config.SAMPLE_SIZE
        },
        "triplets": unique_triplets,
        "test_queries": test_queries
    }

    print("\n>>> Сохранение результатов...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f" Файл сохранён: {output_path}")
    print(f"   Размер: {os.path.getsize(output_path)} байт")
    print("=" * 60)


if __name__ == "__main__":
    main()