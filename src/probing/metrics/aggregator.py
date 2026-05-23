"""Агрегация сравнения baseline и методов редактирования."""

from collections import defaultdict
from typing import Dict, List

from src.probing.metrics.calculator import flatten_metrics
from src.probing.metrics.schema import ComparisonReport, ComparisonRow, ProbingMetrics


def build_comparison_report(
    baseline_by_dataset: Dict[str, ProbingMetrics],
    edited_by_dataset_method: Dict[str, Dict[str, ProbingMetrics]],
) -> ComparisonReport:
    """Строит табличное сравнение baseline vs edited по всем датасетам."""
    report = ComparisonReport()
    method_scores = defaultdict(list)
    dataset_scores = defaultdict(dict)

    for dataset, base_metrics in baseline_by_dataset.items():
        base_flat = flatten_metrics(base_metrics)
        methods = edited_by_dataset_method.get(dataset, {})

        best_method = ""
        best_delta = float("-inf")

        for method, edited_metrics in methods.items():
            edited_flat = flatten_metrics(edited_metrics)
            deltas: List[float] = []

            for metric_name, edited_value in edited_flat.items():
                base_value = base_flat.get(metric_name, 0.0)
                delta = edited_value - base_value
                deltas.append(delta)
                report.rows.append(
                    ComparisonRow(
                        dataset=dataset,
                        method=method,
                        metric_name=f"delta.{metric_name}",
                        value=delta,
                    )
                )

            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            method_scores[method].append(avg_delta)
            dataset_scores[dataset][method] = avg_delta

            if avg_delta > best_delta:
                best_delta = avg_delta
                best_method = method

        report.best_method_per_dataset[dataset] = best_method

    for method, vals in method_scores.items():
        report.summary_by_method[method] = {
            "avg_delta": (sum(vals) / len(vals)) if vals else 0.0,
            "num_datasets": float(len(vals)),
        }

    for dataset, per_method in dataset_scores.items():
        report.summary_by_dataset[dataset] = per_method

    return report
