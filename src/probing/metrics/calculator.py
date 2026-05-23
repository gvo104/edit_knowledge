"""Расчёт и сравнение метрик probing в нормализованном формате."""

from typing import Any, Dict, List

from src.probing.metrics.schema import CategoryMetrics, ProbingMetrics


def compute_metrics_from_results(
    results: Dict[str, List[Dict[str, Any]]],
    good_enough_threshold: float,
) -> ProbingMetrics:
    """Строит нормализованные метрики из сырых результатов probing."""
    categories: Dict[str, CategoryMetrics] = {}

    for query_type, rows in results.items():
        total = len(rows)
        if total == 0:
            categories[query_type] = CategoryMetrics(0, 0, 0.0, 0, 0.0, 0.0)
            continue

        scores = [float(r.get("score", 0.0)) for r in rows]
        perfect = sum(1 for s in scores if s == 1.0)
        good = sum(1 for s in scores if s >= good_enough_threshold)
        mean = sum(scores) / total

        categories[query_type] = CategoryMetrics(
            total=total,
            perfect_1_0=perfect,
            accuracy_1_0=perfect / total,
            good_enough_count=good,
            good_enough_rate=good / total,
            mean_score=mean,
        )

    return ProbingMetrics(categories=categories)


def flatten_metrics(metrics: ProbingMetrics) -> Dict[str, float]:
    """Преобразует объект метрик в плоский словарь key->value."""
    out: Dict[str, float] = {}
    for cat, m in metrics.categories.items():
        out[f"{cat}.accuracy_1_0"] = m.accuracy_1_0
        out[f"{cat}.good_enough_rate"] = m.good_enough_rate
        out[f"{cat}.mean_score"] = m.mean_score
    return out
