from tensorflow.keras.layers import Layer as KerasLayer

from abc import ABC
from abc import abstractmethod
from typing import Sequence
import logging
log = logging.getLogger(__name__)


def _check_num_layers_and_log(call_fn):
    def wrap_call_fn(self, layers):
        assert len(layers) == self.num_layers, \
            f"The number of layers for the {self.__class__} optimizations should be {self.num_layers}. The provided" \
            f" number of layers to the optimizer was {len(layers)}."
        log.info(f"Calling optimization {self.__class__} on layers:{layers}.")
        return call_fn(self, layers)
    return wrap_call_fn


class QKerasOptimization(ABC):
    num_layers: int
    priority: int = 0

    def __init_subclass__(cls):
        """
            We decorate all child classes to check if the length of the tuple matches
            the num_layers variable.
        """
        cls.__call__ = _check_num_layers_and_log(cls.__call__)  # type:ignore

    @_check_num_layers_and_log
    def __call__(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        return self._call_impl(layers)

    @abstractmethod
    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        return []

    @abstractmethod
    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return False
