import qkeras

from tensorflow.keras.layers import Layer as KerasLayer

from typing import Sequence

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasAddInputQuantization(QKerasOptimization):
    """
        We use pythons flexible typing system to add (monkey patch) the input quantization to the layer object.
        We do this so that the transform later has all the information it needs to transform a qkeras layer into
        a lbir layer. This must run after the activation fold operation, so its order number is higher.
    """
    num_layers = 2
    order = 6

    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        layers[1].input_quantizer_internal = layers[0].activation
        if isinstance(layers[0], qkeras.QActivation):
            layers[0].c4ml_remove_layer = True
        return layers

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return ((type(layers[0]) is qkeras.QDense and
                 type(layers[1]) is qkeras.QDense) or
                (type(layers[0]) is qkeras.QActivation and
                 type(layers[1]) is qkeras.QDense))
