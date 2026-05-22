"""
Основной цикл probing: пакетная обработка всех запросов (включая locality).
"""
from typing import Dict, List, Any
from llm.query_engine import get_model_answers_batch, evaluate_answer


def run_probing(test_queries: Dict[str, List[Dict[str, Any]]],
                tokenizer, model,
                gen_kwargs: dict,
                eval_strategy: str = "improved",
                verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Принимает словарь с любыми категориями запросов.
    Возвращает результаты для каждой категории.
    """
    results = {}

    for query_type, items in test_queries.items():
        if not items:
            continue

        print(f"Обработка {query_type} запросов ({len(items)} шт.)...")

        # Извлекаем вопросы и ожидаемые ответы
        questions = [item['question'] for item in items]
        expecteds = [item['expected'] for item in items]

        # Параметры генерации из конфига
        batch_size = gen_kwargs.get('batch_size', 16)
        max_new_tokens = gen_kwargs.get('max_new_tokens', 50)
        do_sample = gen_kwargs.get('do_sample', False)
        temperature = gen_kwargs.get('temperature', 0.1)

        print(f"  Генерация ответов (batch_size={batch_size})...")
        model_answers = get_model_answers_batch(
            questions, tokenizer, model,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature
        )

        # Оценка и сбор результатов
        scores = []
        for item, ans, exp in zip(items, model_answers, expecteds):
            try:
                score = evaluate_answer(ans, exp, strategy=eval_strategy)
            except Exception:
                score = 0.0
            scores.append(score)
            results.setdefault(query_type, []).append({
                "question": item['question'],
                "expected": exp,
                "model_answer": ans,
                "score": score,
                "triplet": item.get('triplet', None)
            })

        avg = sum(scores) / len(scores) if scores else 0
        print(f"  Средний score для {query_type}: {avg:.3f}\n")

    return results