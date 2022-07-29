import qkeras

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.math import rsqrt

from typing import Sequence

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasBNQDenseQReluFuse(QKerasOptimization):
    """
        Fuses the BatchNorm and QDense layer with a quantized_relu activation function.
    """
    num_layers = 3
    order = 2

    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        mm = layers[1].moving_mean
        mv = layers[1].moving_variance
        beta = layers[1].beta
        gamma = layers[1].gamma
        epsilon = layers[1].epsilon
        b = layers[0].bias
        w = layers[0].kernel
        inv = gamma * rsqrt(mv + epsilon)
        layers[0].kernel = inv * w
        layers[0].bias = (inv * (b - mm)) + beta
        layers[1].c4ml_remove_layer = True
        return layers

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return (type(layers[0]) is qkeras.QDense and
                type(layers[1]) is BatchNormalization and
                type(layers[2]) is qkeras.QActivation and
                type(layers[2].activation) is qkeras.quantizers.quantized_relu)
