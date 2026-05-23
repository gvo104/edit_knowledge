from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class Registry:
    """Simple name->config registry for datasets/models/methods/metrics/runs."""
    _items: Dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, value: Any, overwrite: bool = False) -> None:
        if name in self._items and not overwrite:
            raise ValueError(f"Item '{name}' is already registered")
        self._items[name] = value

    def get(self, name: str) -> Any:
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown item '{name}'. Available: {available}")
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def list(self) -> Iterable[str]:
        return sorted(self._items.keys())


class GlobalRegistries:
    """Container for project-wide registries."""

    def __init__(self) -> None:
        self.datasets = Registry()
        self.models = Registry()
        self.methods = Registry()
        self.metrics = Registry()
        self.run_configs = Registry()


global_registries = GlobalRegistries()
