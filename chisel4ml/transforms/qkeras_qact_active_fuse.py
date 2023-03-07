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

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform
from chisel4ml.transforms.qkeras_util import _qact_to_bitwidth
from chisel4ml.transforms.qkeras_util import _qact_to_qtype
from chisel4ml.transforms.qkeras_util import _qact_to_shift
from chisel4ml.transforms.qkeras_util import _qact_to_sign
from chisel4ml.transforms.qkeras_util import _qkeras_base_transform_no_inp


@register_qkeras_transform
class QKerasQActActiveFuse(QKerasTransform):
    """Takes the sequence: QActivation, QDense/QConv and transforms it to a lbir.Layer.
    A QActivation before QDense is needed to determine the input quantization. This
    transform comes after running a fuse qdense, Qact sequence transform, thus
    qdense.activation should be populated. This transform should only take effect at
    the input (so first two layers).
    """

    num_layers = 2
    order = 3

    def _call_impl(self, layers):
        act_shape = layers[0].get_output_shape_at(0)[1:]
        if len(act_shape) == 1:
            input_shape = (1, 1, 1) + act_shape  # qdense
        else:
            input_shape = (1,) + act_shape  # qconv
        assert len(input_shape) == 4
        input_tensor = lbir.QTensor(
            dtype=lbir.Datatype(
                quantization=_qact_to_qtype(layers[0].activation),
                signed=_qact_to_sign(layers[0].activation),
                bitwidth=_qact_to_bitwidth(layers[0].activation),
                shift=_qact_to_shift(
                    layers[0].activation, layers[0].get_output_shape_at(0)[1:]
                ),
                offset=[0],
            ),
            shape=input_shape,  # 1st arg for nodes, 2nd batch dims
        )
        lbir_layer = _qkeras_base_transform_no_inp(layers[1])
        lbir_layer.input.CopyFrom(input_tensor)
        return [lbir_layer]

    def is_applicable(self, layers) -> bool:
        return isinstance(layers[0], qkeras.QActivation) and isinstance(
            layers[1], (qkeras.QDense, qkeras.QConv2D)
        )
