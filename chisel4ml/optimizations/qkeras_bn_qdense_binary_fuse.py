import tensorflow as tf
import qkeras

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.layers import BatchNormalization
from typing import List

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization([qkeras.QDense])
class QKerasBnQdenseBinaryFuse(QKerasOptimization):
    num_layers = 3

    def __call__(self, layers: List[KerasLayer]) -> List[KerasLayer]:
        if self.is_applicable(layers):
            moving_mean = layers[1].moving_mean
            moving_variance = layers[1].moving_variance
            beta = layers[1].beta
            gamma = layers[1].gamma
            bias = layers[0].bias
            return layers
        else:
            return layers

    def is_applicable(self, layers: List[KerasLayer]) -> bool:
        return (type(layers[0]) is qkeras.QDense and
               type(layers[1]) is BatchNormalization and
               type(layers[2]) is qkeras.QActivation and
               type(layers[2].activation) is qkeras.quantizers.binary)
