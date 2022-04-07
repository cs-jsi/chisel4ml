import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer

from typing import List

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_transform


@register_qkeras_transform([tf.keras.layers.Dropout,
                            tf.keras.layers.InputLayer])
class QKerasRemoveDeadLayersOptimization(QKerasOptimization):
    num_layers = 1

    def __call__(self, layers : List[KerasLayer]) -> List[KerasLayer]:
        return []


