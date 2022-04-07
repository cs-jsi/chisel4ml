import tensorflow as tf

from chisel4ml.transforms.transform import KerasLbirTransform
from chisel4ml.transforms import register_keras_transform

import chisel4ml.lbir_python.lbir_pb2 as lbir


@register_keras_transform("keras_remove_dead_layers")
class KerasRemoveDeadLayersTransform(KerasLbirTransform):
    def __init__(self):
        self._dead_layers = ["Dropout",
                             "InputLayer"]

    def is_applicable(self, layer: tf.keras.Layer) -> lbir.Layer:
        return layer.__name__ in self._dead_layers

    def __call__(self, layer: tf.keras.Layer) -> lbir.Layer:
        return ()
