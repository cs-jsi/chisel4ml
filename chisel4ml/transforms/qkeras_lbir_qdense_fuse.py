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
import tensorflow as tf

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform
from chisel4ml.transforms.qkeras_util import _qkeras_base_transform_no_inp


@register_qkeras_transform
class QKerasLbirQDenseFuse(QKerasTransform):
    """Takes the sequeunce: LbirLayer, QDense (with QActivation). And outputs a
    Sequence of two lbir layers.
    """

    num_layers = 2
    order = 5

    def _call_impl(self, layers):
        if isinstance(layers[1], (qkeras.QDense, qkeras.QConv2D)):
            lbir_layer = _qkeras_base_transform_no_inp(layers[1])
            l1_attr = lbir_layer.WhichOneof('sealed_value_optional')
            l0_attr = layers[0].WhichOneof('sealed_value_optional')
            getattr(lbir_layer, l1_attr).input.CopyFrom(getattr(layers[0], l0_attr).output)
        else:
            tf_shape = layers[1].get_output_shape_at(0)[1:]
            lbir_layer = lbir.LayerWrap(lbir.MaxPool2DConfig(
                input=layers[0].output,
                output=QTensor(
                    dtype=layers[0].output.dtype,
                    shape=(tf_shape[2],)
                    + tf_shape[0:2],  # tensorflows HWC -> to LBIR CHW
                ),
            ))
        return [layers[0], lbir_layer]

    def is_applicable(self, layers) -> bool:
        return hasattr(layers[0], "__module__") and layers[0].__module__ == "lbir_pb2" and isinstance(
            layers[1], (qkeras.QDense, qkeras.QConv2D, tf.keras.layers.MaxPooling2D)
        )
