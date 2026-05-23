"""Схемы и типы метрик probing в едином формате."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CategoryMetrics:
    """Метрики для одной категории запросов (direct/inverse/paraphrase/locality)."""

    total: int
    perfect_1_0: int
    accuracy_1_0: float
    good_enough_count: int
    good_enough_rate: float
    mean_score: float


@dataclass
class ProbingMetrics:
    """Нормализованный контейнер метрик probing."""

    categories: Dict[str, CategoryMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categories": {k: asdict(v) for k, v in self.categories.items()}
        }


@dataclass
class ComparisonRow:
    """Строка таблицы сравнения для final report."""

    dataset: str
    method: str
    metric_name: str
    value: float


@dataclass
class ComparisonReport:
    """Структура итогового сравнения baseline vs edited."""

    rows: List[ComparisonRow] = field(default_factory=list)
    summary_by_method: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary_by_dataset: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_method_per_dataset: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [asdict(r) for r in self.rows],
            "summary_by_method": self.summary_by_method,
            "summary_by_dataset": self.summary_by_dataset,
            "best_method_per_dataset": self.best_method_per_dataset,
        }
