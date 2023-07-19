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
from typing import Sequence

import qkeras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.models import Model as KerasModel
from tensorflow.math import rsqrt
from tf2kerassurgeon.operations import delete_layer

from chisel4ml.optimizations import register_qkeras_optimization
from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization


@register_qkeras_optimization
class QKerasBNQDenseFuse(QKerasOptimization):
    """
    Fuses BatchNorm and QDense/QConv layers with a quantized_relu activation function.
    """

    num_layers = 2
    order = 3

    def _call_impl(self, model: KerasModel, layers: Sequence[KerasLayer]) -> KerasModel:
        mm = layers[1].moving_mean
        mv = layers[1].moving_variance
        beta = layers[1].beta
        gamma = layers[1].gamma
        epsilon = layers[1].epsilon
        b = layers[0].bias if layers[0].use_bias else 0
        w = layers[0].kernel
        inv = gamma * rsqrt(mv + epsilon)
        layers[0].kernel.assign(w * inv)
        if not layers[0].use_bias:
            layers[0].use_bias = True
            if isinstance(layers[0], qkeras.QDense):
                bias_shape = layers[0].units
            else:
                bias_shape = layers[0].filters
            layers[0].bias = layers[0].add_weight(
                "bias",
                shape=(bias_shape,),
                dtype=layers[0].dtype,
                trainable=True,
            )
            layers[0].build(input_shape=layers[0].input_shape[1:])
        layers[0].bias.assign(((b - mm) * inv) + beta)
        return delete_layer(model, layers[1], copy=False)

    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return isinstance(layers[0], (qkeras.QConv2D, qkeras.QDense)) and isinstance(
            layers[1], BatchNormalization
        )
