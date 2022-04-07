import tensorflow as tf

from abc import ABC
from abc import abstractmethod

import chisel4ml.lbir_python.lbir_pb2 as lbir


class KerasLbirTransform(ABC):
    num_layers: int

    @abstractmethod
    def __call__(self, layers: tf.keras.Layer) -> lbir.Layer:
        return

    @abstractmethod
    def is_applicable(self, layers: tf.keras.Layer) -> lbir.Layer:
        """
            TODO: remove this and replace with a registration mechanism that
            uses the tf.keras.Layer as a key for the tranform.
                i.e.
                    for keras_layer in keras_model:
                        lbir_model.append(keras.layer.transform_lbir)
        """
        return
