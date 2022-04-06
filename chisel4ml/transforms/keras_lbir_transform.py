import tensorflow as tf

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import chisel4ml.LBIR_pb2 as lbir

class KerasLbirTransform(ABC):
    num_layers : int

    def __init_subclass__(cls, **kwargs):
        """
            We decorate all child classes to check if the length of the tuple matches
            the num_layers variable.
        """
        super().__init_subclass__(**kwargs)
        cls.is_applicable = check_num_layer(cls.is_applicable)

    @abstractmethod
    def __call__(self, layers : tf.keras.Layer) -> lbir.Layer:
        return 

    @abstractmethod
    def is_applicable(self, layers : tf.keras.Layer) -> lbir.Layer:
        """
            TODO: remove this and replace with a registration mechanism that
            uses the tf.keras.Layer as a key for the tranform.
                i.e.
                    for keras_layer in keras_model:
                        lbir_model.append(keras.layer.transform_lbir)
        """
        return


