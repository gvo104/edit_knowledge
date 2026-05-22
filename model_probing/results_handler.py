"""
Сохранение результатов probing и вычисление сводной статистики.
"""
import json
import os
from typing import Dict, List, Any

def compute_summary(results: Dict[str, List[Dict[str, Any]]],
                    good_enough_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Рассчитывает метрики для каждого типа запросов.
    Возвращает словарь summary.
    """
    summary = {}
    for query_type, items in results.items():
        scores = [r['score'] for r in items]
        total = len(scores)
        if total == 0:
            continue
        perfect = sum(1 for s in scores if s == 1.0)
        partial_08 = sum(1 for s in scores if s == 0.8)
        partial_06 = sum(1 for s in scores if s == 0.6)
        wrong = sum(1 for s in scores if s == 0.0)
        good = sum(1 for s in scores if s >= good_enough_threshold)
        summary[query_type] = {
            "total": total,
            "perfect_1.0": perfect,
            "partial_0.8": partial_08,
            "partial_0.6": partial_06,
            "wrong_0.0": wrong,
            "accuracy_1.0": perfect / total if total else 0,
            "mean_score": sum(scores) / total if total else 0,
            "good_enough_count": good,
            "good_enough_rate": good / total if total else 0
        }
    return summary


def print_summary(summary: Dict[str, Any]):
    """Выводит сводные метрики в консоль."""
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ KNOWLEDGE PROBING")
    print("=" * 60)
    for qtype, metrics in summary.items():
        print(f"\n{qtype.upper()}:")
        print(f"  Всего запросов: {metrics['total']}")
        print(f"  Perfect (1.0): {metrics['perfect_1.0']} ({metrics['accuracy_1.0']*100:.1f}%)")
        print(f"  Good enough (≥0.8): {metrics['good_enough_count']} ({metrics['good_enough_rate']*100:.1f}%)")
        print(f"  Mean score: {metrics['mean_score']:.3f}")


def save_results(results: Dict[str, List[Dict[str, Any]]],
                 summary: Dict[str, Any],
                 model_name: str,
                 device: str,
                 dtype: str,
                 output_path: str) -> None:
    """Сохраняет полные результаты probing в JSON."""
    output_obj = {
        "model_info": {
            "name": model_name,
            "device": device,
            "dtype": dtype
        },
        "summary": summary,
        "detailed_results": results
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    print(f"\n Результаты сохранены: {output_path}")


def print_sample_answers(results: Dict[str, List[Dict[str, Any]]], n: int = 3):
    """Выводит примеры ответов модели для каждого типа запросов."""
    print("\n" + "=" * 60)
    print("ПРИМЕРЫ ОТВЕТОВ МОДЕЛИ:")
    print("=" * 60)
    for qtype, items in results.items():
        if not items:
            continue
        print(f"\n{qtype.upper()} (первые {n} примеров):")
        for i, r in enumerate(items[:n]):
            print(f"  {i+1}. Q: {r['question']}")
            print(f"     Expected: {r['expected']}")
            # Обрезаем ответ, чтобы не засорять консоль
            ans = r['model_answer'][:100].replace('\n', ' ')
            print(f"     Model: {ans}")
            print(f"     Score: {r['score']}")