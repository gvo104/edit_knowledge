import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from llm.model_loader import load_model_and_tokenizer   # ★ изменённый импорт
from model_probing.probing_runner import run_probing    # локальный импорт
from model_probing.results_handler import compute_summary, print_summary, save_results, print_sample_answers

def main():
    print("=" * 60)
    print("PROBING МОДЕЛИ: Qwen2.5-0.5B-Instruct на gene2pubtatorcentral")
    print("=" * 60)

    triplets_path = Path(config.TRIPLETS_PATH)
    with open(triplets_path, 'r', encoding='utf-8') as f:
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

    output_path = Path(config.PROBING_DIR) / config.PROBING_OUTPUT_FILENAME
    save_results(results, summary, config.MODEL_NAME,
                 str(model.device), str(model.dtype), str(output_path))
    print("Готово.")

if __name__ == "__main__":
    main()