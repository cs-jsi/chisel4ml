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
import math

import qkeras

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.lbir.datatype_pb2 import Datatype
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform
from chisel4ml.transforms.qkeras_util import _qact_to_bitwidth
from chisel4ml.transforms.qkeras_util import _qact_to_sign


@register_qkeras_transform
class QKerasAudioPreprocess(QKerasTransform):
    """
    Transforms an audio preprocess layer into the LBIR representation.
    """

    num_layers = 2
    order = 2

    def _call_impl(self, layers):
        bitwidth = _qact_to_bitwidth(layers[0].activation)
        signed = _qact_to_sign(layers[0].activation)
        assert bitwidth == 12
        assert signed
        fft_config = layers[1].cfg
        fft_config.input.CopyFrom(
            QTensor(
                dtype=Datatype(
                    quantization=Datatype.QuantizationType.UNIFORM,
                    signed=True,
                    bitwidth=12,
                    shift=[0],
                    offset=[0],
                ),
                shape=[
                    layers[1].cfg.num_frames,
                    layers[1].cfg.fft_size,
                ],  # KERNEL, CH, WIDTH, HEIGHT
            )
        )
        fft_config.output.CopyFrom(
            QTensor(
                dtype=Datatype(
                    quantization=Datatype.QuantizationType.UNIFORM,
                    signed=True,
                    bitwidth=int(24 + math.log2(layers[1].cfg.fft_size)),
                    shift=[12],
                    offset=[0],
                ),
                shape=[
                    layers[1].cfg.num_frames,
                    layers[1].cfg.fft_size,
                ],  # KERNEL, CH, WIDTH, HEIGHT
            )
        )
        return [lbir.LayerWrap(fft=fft_config)]

    def is_applicable(self, layers) -> bool:
        return isinstance(layers[0], qkeras.QActivation) and isinstance(
            layers[1], FFTLayer
        )
