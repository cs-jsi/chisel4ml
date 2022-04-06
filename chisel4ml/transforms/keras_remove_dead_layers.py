import tensorflow as tf

from typing import Tuple

from transforms.transform import KerasLbirTransform
import LBIR_pb2 as lbir


@register_keras_transform("keras_remove_dead_layers")
class KerasRemoveDeadLayersTransform(KerasLbirTransform):
    def __init__(self):
        num_layers = 1

        _dead_layers = ["Dropout", 
                        "InputLayer"]

    def is_applicable(self, layers : Tuple[tf.keras.Layer, ...]) -> lbir:Layer:
        return layers[0].__name__ in _dead_layers

    def __call__(self, layers : Tuple[tf.keras.Layer]) -> lbir.Layer:
        return ()


