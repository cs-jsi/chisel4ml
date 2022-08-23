import qkeras

from tensorflow.keras.layers import Layer as KerasLayer

from typing import Sequence

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasRemoveActivationLayer(QKerasOptimization):
    """
        We remove activation layers by monkey patching layers around it.
    """
    num_layers = 3
    order = 5

    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        layers[0].activation = layers[1].activation
        layers[2].input_quantizer_internal = layers[1].activation
        layers[1].c4ml_remove_layer = True
        return layers

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return (type(layers[0]) is qkeras.QDense and
                type(layers[1]) is qkeras.QActivation and
                type(layers[2]) is qkeras.QDense)
