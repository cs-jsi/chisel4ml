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
from functools import reduce

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform


@register_qkeras_transform
class LbirFlattenShapeConvDense(QKerasTransform):
    """Takes the sequeunce: LbirLayer.Type.Conv2D, Lbir.Layer.type.Dense And flattens a
    the input shape of Dense. Note that this is just flattening of the shape values, no
    memory rearangement is actually done.
    """

    num_layers = 2
    order = 7

    def _call_impl(self, layers):
        if layers[0].HasField("conv2d"):
            layers[1].dense.input.shape[:] = [
                reduce(lambda x, y: x * y, layers[0].conv2d.output.shape)
            ]
        else:
            layers[1].dense.input.shape[:] = [
                reduce(lambda x, y: x * y, layers[0].maxpool2d.output.shape)
            ]
        return layers

    def is_applicable(self, layers) -> bool:
        return (
            isinstance(layers[0], lbir.LayerWrap)
            and (layers[0].HasField("conv2d") or layers[0].HasField("maxpool2d"))
            and isinstance(layers[1], lbir.LayerWrap)
            and layers[1].HasField("dense")
            and len(layers[1].dense.input.shape) > 1  # prevents endless application
        )
