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

from chisel4ml.transforms.qkeras_transforms import QKerasTransform
from chisel4ml.transforms import register_qkeras_transform
import chisel4ml.lbir.lbir_pb2 as lbir


@register_qkeras_transform
class QKerasQDenseQActFuse(QKerasTransform):
    """
        Takes the sequence: QDense, QActivations and merges the QActivation to the QDense.activation parameter. 
        This transform simplifies further transformations.
    """
    num_layers = 2
    order = 1

    def _call_impl(self, layers):
        # linear is a function, not a class (hence the is)
        assert (layers[0].activation is None) or (layers[0].activation is linear), \
                f"""The QDense layer {layer} has an activation that is not linear, and is followed by a different 
                    activation. The activation in qdense is {layers[0].activation} and the sepereate activation is 
                    {layers[1]}."""
        layers[0].activation = layers[1]
        return [layers[0]]
        
    def is_applicable(self, layers) -> bool:
        return (isinstance(layers[0], qkeras.QDense) and
                isinstance(layers[1], qkeras.QActivation))
