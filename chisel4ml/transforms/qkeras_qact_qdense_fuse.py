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

from chisel4ml.transforms.qkeras_transforms import QKerasTransform
from chisel4ml.transforms import register_qkeras_transform
import chisel4ml.lbir.lbir_pb2 as lbir


@register_qkeras_transform
class QKerasQActQDenseFuse(QKerasTransform):
    """
        Takes the sequence: QActivation, QDense and transforms it to a lbir.Layer. A QActivation before QDense is 
        needed to determine the input quantization. This transform comes after running a fuse qdense, Qact sequence
        transform, thus qdense.activation should be populated. This transform should only take effect at the input 
        (so first two layers). 
    """
    num_layers = 2
    order = 2

    def _call_impl(self, layers):
        lbir_layer = lbir.Layer()
        lbir_layer = qkeras_base_transform(layers[1])
        lbir_layer = qkeras_add_input_tensor(layers[0])
        return [lbir_layer]
        
    def is_applicable(self, layers) -> bool:
        return (isinstance(layers[0], qkeras.QActivation) and
                isinstance(layers[1], qkeras.QDense))
