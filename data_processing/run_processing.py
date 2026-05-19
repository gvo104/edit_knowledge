"""
Основной скрипт обработки данных.
Запускает всю цепочку: статистика → выборка → триплеты → тестовые запросы → сохранение.
"""
import sys
import json
import os
import argparse
from pathlib import Path

# === Настройка путей для импорта общего конфига ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # Теперь импортируется из корня проекта

# Локальные модули обработки данных (лежат в этой же папке)
from data_reader import iter_records
from stats import compute_statistics
from triplet_extractor import extract_sample, group_entities_by_pmid, build_triplets
from query_generator import generate_queries


def main():
    parser = argparse.ArgumentParser(description="Обработка PubTator датасета в триплеты")
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME,
                        help='Имя датасета: gene, disease, mutation')
    args = parser.parse_args()
    dataset_name = args.dataset
    ds_config = config.get_dataset_config(dataset_name)

    data_path = ds_config['path']
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === ВЫВОД ЗАГОЛОВКА ===
    print("=" * 60)
    print(f"ОБРАБОТКА ДАННЫХ: извлечение SPO-триплетов из {dataset_name}2pubtatorcentral")
    print("=" * 60)

    # 2. Первый проход – статистика (потоковый)
    print("\n>>> Этап 1: Сбор статистики по всему файлу...")
    stats = compute_statistics(iter_records(data_path), verbose=config.VERBOSE)
    print(f"    Всего записей: {stats['total_lines']}")
    print(f"    Уникальных PMID: {stats['unique_pmids']}")
    print(f"    Уникальных сущностей: {stats['unique_entities']}")

    # 3. Второй проход – выборка первых SAMPLE_SIZE уникальных PMID
    print(f"\n>>> Этап 2: Загрузка выборки из первых {config.SAMPLE_SIZE} статей...")
    sample_records = extract_sample(iter_records(data_path), config.SAMPLE_SIZE)
    print(f"    Записей в выборке: {len(sample_records)}")

    # 4. Группировка сущностей по PMID
    print("\n>>> Этап 3: Группировка сущностей по статьям...")
    entities_by_pmid = group_entities_by_pmid(sample_records)
    print(f"    Статей в выборке: {len(entities_by_pmid)}")

    # 5. Построение триплетов
    print("\n>>> Этап 4: Создание SPO-триплетов...")
    triplets = build_triplets(
        entities_by_pmid,
        config.MAX_COOCCUR_GENES,
        predicate="mentioned in",
        cooccur_predicate="co-occurs with"
    )
    unique_triplets = list(triplets)  # для сериализации превращаем в список
    print(f"    Уникальных триплетов: {len(unique_triplets)}")

    # 6. Генерация тестовых запросов
    print("\n>>> Этап 5: Генерация тестовых запросов...")
    test_queries = generate_queries(triplets, entity_type=ds_config['entity_type'])
    for key in test_queries:
        print(f"    {key}: {len(test_queries[key])} шт.")

    # 7. Сохранение результата с версией
    version = config.VERSION_TIMESTAMP
    output_filename = config.OUTPUT_FILENAME_TEMPLATE.format(
        dataset=dataset_name, version=version
    )
    output_path = output_dir / output_filename

    output = {
        "source": f"{dataset_name}2pubtatorcentral",
        "dataset": dataset_name,
        "version": version,
        "entity_type": ds_config['entity_type'],
        "sample_info": {
            "total_pmids": len(entities_by_pmid),
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

    print(f"Файл сохранён: {output_path}")
    print(f"   Размер: {os.path.getsize(output_path)} байт")
    print("=" * 60)


if __name__ == "__main__":
    main()