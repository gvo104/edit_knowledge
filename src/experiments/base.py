from abc import ABC, abstractmethod
from typing import Optional

from src.core.run_context import RunContext


class ExperimentInterface(ABC):
    @abstractmethod
    def run_prepare_data(self, **kwargs) -> RunContext:
        raise NotImplementedError

    @abstractmethod
    def run_baseline_probe(self, version: Optional[str] = None, **kwargs) -> RunContext:
        raise NotImplementedError

    @abstractmethod
    def run_train_edit_method(self, version: Optional[str] = None, **kwargs) -> RunContext:
        raise NotImplementedError

    @abstractmethod
    def run_post_edit_probe(self, version: Optional[str] = None, **kwargs) -> RunContext:
        raise NotImplementedError

    @abstractmethod
    def run_aggregate_results(self, **kwargs) -> RunContext:
        raise NotImplementedError
