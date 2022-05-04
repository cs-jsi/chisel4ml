import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer

from typing import Sequence

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasRemoveDeadLayersOptimization(QKerasOptimization):
    num_layers = 1
    priority = 1

    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        layers[0].c4ml_remove_layer = True
        return layers

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return (type(layers[0]) is tf.keras.layers.Dropout or
                type(layers[0]) is tf.keras.layers.InputLayer)
