import qkeras

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.math import sqrt
from tensorflow.math import multiply as mul
from tensorflow.math import truediv as div

from typing import List
import copy

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasBNQDenseBinaryFuse(QKerasOptimization):
    """ 
        Fuses the BatchNorm and QDense layer with a binary quantizer. For more information read the paper
        on Binarized Neural networks by Courbariaux and Hubara et al.: https://arxiv.org/pdf/1602.02830.pdf. 
    """
    num_layers = 3

    def __call__(self, layers: List[KerasLayer]) -> List[KerasLayer]:
        new_layers = copy.deepcopy(layers)
        mm = layers[1].moving_mean
        mv = layers[1].moving_variance
        b = layers[1].beta
        g = layers[1].gamma
        b = layers[0].bias
        new_layers[0].bias = (mm - b) - div(mul(sqrt(mv), b), g)
        del new_layers[1]
        return new_layers

    def is_applicable(self, layers: List[KerasLayer]) -> bool:
        return (type(layers[0]) is qkeras.QDense and
                type(layers[1]) is BatchNormalization and
                type(layers[2]) is qkeras.QActivation and
                type(layers[2].activation) is qkeras.quantizers.binary)
