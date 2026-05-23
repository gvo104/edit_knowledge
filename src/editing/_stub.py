from typing import Any, Dict

from src.editing.base import EditingMethod


class StubEditingMethod(EditingMethod):
    """Placeholder for not-yet-implemented editing methods."""

    name = "stub"

    def prepare(self, **kwargs) -> Dict[str, Any]:
        return {"status": "prepared", "implemented": False, "params": kwargs}

    def train_or_edit(self, **kwargs) -> Dict[str, Any]:
        return {
            "status": "skipped",
            "implemented": False,
            "reason": "Method adapter is a placeholder",
            "params": kwargs,
        }

    def save(self, **kwargs) -> Dict[str, Any]:
        return {"status": "noop", "implemented": False}

    def load(self, **kwargs) -> Dict[str, Any]:
        return {"status": "noop", "implemented": False}

    def apply_to_model(self, **kwargs) -> Dict[str, Any]:
        return {"status": "noop", "implemented": False}
