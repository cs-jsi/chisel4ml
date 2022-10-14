# Copyright 2022 Computer Systems Department, Jozef Stefan Insitute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#  https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import qkeras
import numpy as np

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.math import sqrt
from tensorflow.math import multiply as mul
from tensorflow.math import truediv as div

from typing import Sequence

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasBNQDenseBinaryFuse(QKerasOptimization):
    """
        Fuses the BatchNorm and QDense layer with a binary quantizer. For more information read the paper
        on Binarized Neural networks by Umuroglu et al.: https://arxiv.org/pdf/1612.07119.pdf.
    """
    num_layers = 3
    order = 2

    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        mm = layers[1].moving_mean
        mv = layers[1].moving_variance
        beta = layers[1].beta
        gamma = layers[1].gamma
        epsilon = layers[1].epsilon
        assert np.amin(gamma) > 0
        b = layers[0].bias
        thresh = (mm - b) - div(mul(sqrt(mv + epsilon), beta), gamma)
        layers[0].bias = -thresh
        layers[1].c4ml_remove_layer = True
        return layers

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return (type(layers[0]) is qkeras.QDense and
                type(layers[1]) is BatchNormalization and
                type(layers[2]) is qkeras.QActivation and
                type(layers[2].activation) is qkeras.quantizers.binary)
