import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer

from abc import ABC
from abc import abstractmethod
from typing import List, Type

def _check_num_layers(call_fn):
    def wrap_call_fn(self, layers : List[KerasLayer]) -> List[KerasLayer]:
        assert len(layers) == num_layers;
        return call_fn()
    return wrap_call_fn

class QKerasOptimization(ABC):
    num_layers : int

    #def __init_subclass__(cls, **kwargs):
    #    """
    #        We decorate all child classes to check if the length of the tuple matches
    #        the num_layers variable.
    #    """
    #    super().__init_subclass__(**kwargs)
    #    cls.__call__ = _check_num_layers(cls.__call__)

    @abstractmethod
    @_check_num_layers
    def __call__(self, layers : List[KerasLayer]) -> List[KerasLayer]:
        return



