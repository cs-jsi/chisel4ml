from tensorflow.keras.layers import Layer as KerasLayer

from abc import ABC
from abc import abstractmethod
from typing import Sequence
import logging
log = logging.getLogger(__name__)


class QKerasOptimization(ABC):
    num_layers: int
    order: int = 0

    def __call__(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        assert len(layers) == self.num_layers, \
            f"The number of layers for the {self.__class__} optimizations should be {self.num_layers}. The provided" \
            f" number of layers to the optimizer was {len(layers)}."
        log.info(f"Calling optimization {self.__class__} on layers:{layers}.")
        return self._call_impl(layers)

    @abstractmethod
    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        return []

    @abstractmethod
    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return False
