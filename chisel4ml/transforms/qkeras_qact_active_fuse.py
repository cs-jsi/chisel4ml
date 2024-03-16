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
from tensorflow.keras.layers import MaxPooling2D

from chisel4ml.lbir.datatype_pb2 import Datatype
from chisel4ml.lbir.lbir_pb2 import LayerWrap
from chisel4ml.lbir.lbir_pb2 import MaxPool2DConfig
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.qkeras_extensions import MaxPool2dCF
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform
from chisel4ml.transforms.qkeras_util import _qact_to_bitwidth
from chisel4ml.transforms.qkeras_util import _qact_to_qtype
from chisel4ml.transforms.qkeras_util import _qact_to_shift
from chisel4ml.transforms.qkeras_util import _qact_to_sign
from chisel4ml.transforms.qkeras_util import _qkeras_base_transform_no_inp


@register_qkeras_transform
class QKerasQActActiveFuse(QKerasTransform):
    """Takes the sequence: QActivation, QDense and transforms it to a lbir.Layer. A
    QActivation before QDense is needed to determine the input quantization. This
    transform comes after running a fuse qdense, Qact sequence transform, thus
    qdense.activation should be populated. This transform should only take effect at
    the input (so first two layers).
    """

    num_layers = 2
    order = 5

    def _call_impl(self, layers):
        shape = layers[0].get_output_shape_at(0)[1:]
        if isinstance(
            layers[1],
            (qkeras.QConv2D, QDepthwiseConv2DPermuted, MaxPooling2D, MaxPool2dCF),
        ):
            if layers[1].data_format == "channels_last":
                shape = [shape[2]] + [*shape[0:2]]
        input_tensor = QTensor(
            dtype=Datatype(
                quantization=_qact_to_qtype(layers[0].activation),
                signed=_qact_to_sign(layers[0].activation),
                bitwidth=_qact_to_bitwidth(layers[0].activation),
                shift=_qact_to_shift(layers[0].activation, [1]),
                offset=[0],
            ),
            shape=shape,
        )
        if isinstance(layers[1], MaxPooling2D):
            oshape = layers[1].get_output_shape_at(0)[1:]
            if layers[1].data_format == "channels_last":
                oshape = [oshape[2]] + [*oshape[0:2]]
            output_tensor = QTensor(dtype=input_tensor.dtype, shape=oshape)
            lbir_layer = LayerWrap(
                maxpool2d=MaxPool2DConfig(input=input_tensor, output=output_tensor)
            )
        elif isinstance(layers[1], MaxPool2dCF):
            oshape = (shape[0],) + tuple(
                map(lambda x: x // layers[1].pool_size[0], shape[1:])
            )
            assert layers[1].pool_size[0] == layers[1].pool_size[1]
            assert oshape[1:] == tuple(
                map(lambda x: x / layers[1].pool_size[0], shape[1:])
            )
            assert oshape[0] == shape[0]
            output_tensor = QTensor(dtype=input_tensor.dtype, shape=oshape)
            lbir_layer = LayerWrap(
                maxpool2d=MaxPool2DConfig(input=input_tensor, output=output_tensor)
            )
        else:
            lbir_layer = _qkeras_base_transform_no_inp(layers[1])

        if lbir_layer.HasField("dense"):
            lbir_layer.dense.input.CopyFrom(input_tensor)
        elif lbir_layer.HasField("conv2d"):
            lbir_layer.conv2d.input.CopyFrom(input_tensor)
        elif lbir_layer.HasField("maxpool2d"):
            return [lbir_layer]
        else:
            raise Exception("lbir_layer should have either dense or conv2d field set.")

        return [lbir_layer]

    def is_applicable(self, layers) -> bool:
        return isinstance(layers[0], qkeras.QActivation) and isinstance(
            layers[1],
            (
                qkeras.QDense,
                qkeras.QConv2D,
                QDepthwiseConv2DPermuted,
                MaxPooling2D,
                MaxPool2dCF,
            ),
        )
