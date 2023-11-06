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
from tensorflow.keras.activations import linear
from tensorflow.keras.layers import Activation

from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform


@register_qkeras_transform
class QKerasActiveQActFuse(QKerasTransform):
    """
    Takes the sequence: QDense (or QConv2D), QActivations and merges the QActivation to
    the QDense/QConv2D  activation parameter. This transform simplifies further
    transformations.
    """

    num_layers = 2
    order = 4

    def _call_impl(self, layers):
        layers[0].activation = layers[1].activation
        return [layers[0]]

    def is_applicable(self, layers) -> bool:
        return (
            isinstance(
                layers[0], (qkeras.QDense, qkeras.QConv2D, QDepthwiseConv2DPermuted)
            )
            and (layers[0].activation is None or layers[0].activation is linear)
            and isinstance(layers[1], (qkeras.QActivation, Activation))
        )
