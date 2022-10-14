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

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.activations import linear

from typing import Sequence

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from chisel4ml.optimizations import register_qkeras_optimization


@register_qkeras_optimization
class QKerasActivationFold(QKerasOptimization):
    """
        In keras the activation can be either as a seperate layer or as part of active layer (i.e. QDense). For the
        sake of simplifying the rest of the code, we fold the activation layer in to the active layer, by setting its
        activation variable.
    """
    num_layers = 2
    order = 4

    def _call_impl(self, layers: Sequence[KerasLayer]) -> Sequence[KerasLayer]:
        layers[0].activation = layers[1].activation
        layers[1].c4ml_remove_layer = True
        return layers

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return (type(layers[0]) is qkeras.QDense and
                layers[0].activation is linear and
                type(layers[1]) is qkeras.QActivation)
