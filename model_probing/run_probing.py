"""
Запуск probing модели на сгенерированных триплетах.
Поддерживает загрузку LoRA-адаптера через --lora_path.
При наличии адаптера результаты сохраняются в отдельную папку и сравниваются с baseline.
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

def _load_baseline_summary(dataset_name: str):
    """Ищет последний baseline probing (без _lora) и возвращает его summary или None."""
    baseline_dir = Path(config.PROBING_DIR)
    pattern = str(baseline_dir / f"*_{dataset_name}_*_probing_results.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("summary")
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Knowledge probing на триплетах")
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME,
                        help='Имя датасета: gene, disease, mutation')
    parser.add_argument('--version', type=str, default=None,
                        help='Версия файла триплетов (например, 20250120_143022). Если не указана, используется последняя.')
    parser.add_argument('--force', action='store_true',
                        help='Перезаписать существующие результаты probing')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Путь к адаптеру LoRA (если указан, модель загружается с ним)')
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
    if args.lora_path:
        print(f"LoRA адаптер: {args.lora_path}")
    print("=" * 60)

    with open(triplets_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_queries = data['test_queries']
    # Добавляем locality, если есть
    if 'locality_queries' in data:
        test_queries['locality'] = data['locality_queries']
    print(f"Загружено тестовых запросов: { {k: len(v) for k, v in test_queries.items()} }")

    tokenizer, model = load_model_and_tokenizer(
        config.MODEL_NAME,
        torch_dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE
    )

    # Загрузка LoRA адаптера, если указан
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"LoRA адаптер загружен из {args.lora_path}")

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

    # --- Сохранение ---
    # Если используется LoRA, сохраняем в отдельную папку
    output_dir = config.PROBING_DIR_LORA if args.lora_path else config.PROBING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = config.MODEL_NAME.split('/')[-1].replace('-Instruct', '')
    version = config.VERSION_TIMESTAMP
    output_filename = config.PROBING_OUTPUT_FILENAME_TEMPLATE.format(
        model_short=model_short,
        dataset=dataset_name,
        version=version
    )
    output_path = output_dir / output_filename

    # Перезапись, если указан --force
    if args.force and output_path.exists():
        output_path.unlink()
        print(f"Старый файл результатов удалён: {output_path}")

    save_results(results, summary, config.MODEL_NAME,
                 str(model.device), str(model.dtype), str(output_path))
    print("Готово.")

    # --- Сравнение с baseline (только для LoRA) ---
    if args.lora_path:
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ С BASELINE (без LoRA)")
        print("=" * 60)
        baseline_summary = _load_baseline_summary(dataset_name)
        if baseline_summary is None:
            print("Baseline probing не найден. Сравнение невозможно.")
        else:
            # Выводим таблицу сравнения по всем общим категориям
            categories = sorted(set(list(summary.keys()) + list(baseline_summary.keys())))
            header = f"{'Категория':<15} {'Метрика':<20} {'Baseline':<12} {'После LoRA':<12} {'Изменение'}"
            print(header)
            print("-" * len(header))
            for cat in categories:
                base = baseline_summary.get(cat, {})
                lora = summary.get(cat, {})
                # Сравниваем mean_score
                base_mean = base.get('mean_score', 0)
                lora_mean = lora.get('mean_score', 0)
                diff = lora_mean - base_mean
                print(f"{cat:<15} {'mean_score':<20} {base_mean:<12.4f} {lora_mean:<12.4f} {diff:+.4f}")
                # Сравниваем perfect (accuracy_1.0)
                base_perfect = base.get('accuracy_1.0', 0)
                lora_perfect = lora.get('accuracy_1.0', 0)
                diff_perfect = lora_perfect - base_perfect
                print(f"{'':<15} {'perfect (1.0)':<20} {base_perfect:<12.4f} {lora_perfect:<12.4f} {diff_perfect:+.4f}")
                # good_enough_rate
                base_good = base.get('good_enough_rate', 0)
                lora_good = lora.get('good_enough_rate', 0)
                diff_good = lora_good - base_good
                print(f"{'':<15} {'good (≥0.8)':<20} {base_good:<12.4f} {lora_good:<12.4f} {diff_good:+.4f}")
            print("=" * 60)

if __name__ == "__main__":
    main()