"""
Подготовка SFT-датасета из триплетов.
Из direct, inverse и paraphrase запросов формируются QA-пары.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def create_sft_dataset(dataset_name: str, version: str = None) -> Path:
    """Создаёт датасет в формате сообщений и возвращает путь к нему."""
    triplets_dir = Path(config.OUTPUT_DIR)
    if version:
        triplets_file = triplets_dir / f"{dataset_name}_{version}_triplets.json"
    else:
        import glob
        pattern = str(triplets_dir / f"{dataset_name}_*_triplets.json")
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            raise FileNotFoundError(f"Триплеты для {dataset_name} не найдены")
        triplets_file = Path(files[0])

    print(f"Загрузка триплетов из {triplets_file}")
    with open(triplets_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_queries = data['test_queries']
    messages_list = []
    system_msg = {"role": "system", "content": "You are a helpful biomedical knowledge assistant. Answer concisely."}

    # Преобразуем все категории (direct, inverse, paraphrase) в обучающие примеры
    for category in ['direct', 'inverse', 'paraphrase']:
        for item in test_queries.get(category, []):
            user_msg = {"role": "user", "content": item['question']}
            assistant_msg = {"role": "assistant", "content": item['expected']}
            messages_list.append([system_msg, user_msg, assistant_msg])

    # Сохранение
    training_dir = Path(config.BASE_DIR) / "data" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    output_file = training_dir / f"{dataset_name}_{config.VERSION_TIMESTAMP}_sft.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(messages_list, f, indent=2, ensure_ascii=False)

    print(f"Создано {len(messages_list)} обучающих примеров")
    print(f"Сохранено в {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME)
    parser.add_argument('--version', type=str, default=None)
    args = parser.parse_args()
    create_sft_dataset(args.dataset, args.version)