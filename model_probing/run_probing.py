"""
Запуск probing модели на сгенерированных триплетах.
"""
import json
import sys
import argparse
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from llm.model_loader import load_model_and_tokenizer
from model_probing.probing_runner import run_probing
from model_probing.results_handler import compute_summary, print_summary, save_results, print_sample_answers

def main():
    parser = argparse.ArgumentParser(description="Knowledge probing на триплетах")
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME,
                        help='Имя датасета: gene, disease, mutation')
    parser.add_argument('--version', type=str, default=None,
                        help='Версия файла триплетов (например, 20250120_143022). Если не указана, используется последняя.')
    args = parser.parse_args()

    dataset_name = args.dataset
    ds_config = config.get_dataset_config(dataset_name)

    # Поиск файла триплетов
    if args.version:
        triplets_file = Path(config.OUTPUT_DIR) / f"{dataset_name}_{args.version}_triplets.json"
    else:
        pattern = str(Path(config.OUTPUT_DIR) / f"{dataset_name}_*_triplets.json")
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            raise FileNotFoundError(f"Нет файлов триплетов для датасета {dataset_name}")
        triplets_file = Path(files[0])

    print("=" * 60)
    print(f"PROBING МОДЕЛИ: {config.MODEL_NAME} на датасете {dataset_name}")
    print(f"Файл триплетов: {triplets_file}")
    print("=" * 60)

    with open(triplets_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_queries = data['test_queries']
    print(f"Загружено тестовых запросов: { {k: len(v) for k, v in test_queries.items()} }")

    tokenizer, model = load_model_and_tokenizer(
        config.MODEL_NAME,
        torch_dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE
    )

    gen_kwargs = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "batch_size": config.BATCH_SIZE
    }

    print("\nНачинаем knowledge probing...\n")
    results = run_probing(
        test_queries, tokenizer, model, gen_kwargs,
        eval_strategy=config.EVALUATION_STRATEGY,
        verbose=config.VERBOSE
    )

    summary = compute_summary(results, good_enough_threshold=config.GOOD_ENOUGH_THRESHOLD)
    print_summary(summary)
    print_sample_answers(results)

    # Сохранение результатов с версией
    model_short = config.MODEL_NAME.split('/')[-1].replace('-Instruct', '')
    version = config.VERSION_TIMESTAMP
    output_filename = config.PROBING_OUTPUT_FILENAME_TEMPLATE.format(
        model_short=model_short,
        dataset=dataset_name,
        version=version
    )
    output_path = Path(config.PROBING_DIR) / output_filename
    save_results(results, summary, config.MODEL_NAME,
                 str(model.device), str(model.dtype), str(output_path))
    print("Готово.")

if __name__ == "__main__":
    main()