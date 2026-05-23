from abc import ABC, abstractmethod
from typing import Any, Dict


class EditingMethod(ABC):
    """Unified interface for all knowledge editing methods."""

    @abstractmethod
    def prepare(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train_or_edit(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def apply_to_model(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
