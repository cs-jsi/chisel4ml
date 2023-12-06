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
import tensorflow as tf

from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform


@register_qkeras_transform
class QKerasRemoveDeadLayers(QKerasTransform):
    """Removes unnecassary layers. Note that LBIR assumes that no memory transformation
    happens, and thus in hardware the dimensions of the array can simply be omited."""

    num_layers = 1
    order = 1

    def _call_impl(self, layers):
        return []

    def is_applicable(self, layers) -> bool:
        return isinstance(
            layers[0],
            (
                tf.keras.layers.Dropout,
                tf.keras.layers.InputLayer,
                tf.keras.layers.Flatten,
                FlattenChannelwise,
            ),
        )
