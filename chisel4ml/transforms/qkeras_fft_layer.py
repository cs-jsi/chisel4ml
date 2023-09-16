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
import numpy as np

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
        shape = layers[0].get_output_shape_at(0)[1:]
        bitwidth = _qact_to_bitwidth(layers[0].activation)
        signed = _qact_to_sign(layers[0].activation)
        #assert shape == (512), "Only 32 by 512 frames supported currently"
        assert bitwidth == 12
        assert signed
        fft_config = lbir.FFTConfig(
            fft_size=512,
            num_frames=32,
            win_fn=layers[1].window_fn.tolist(),
            input=QTensor(
                dtype=Datatype(
                    quantization=Datatype.QuantizationType.UNIFORM,
                    signed=True,
                    bitwidth=12,
                    shift=[0],
                    offset=[0],
                    ),
                shape=[32, 512],  # KERNEL, CH, WIDTH, HEIGHT
                ),
            output=QTensor(
                dtype=Datatype(
                    quantization=Datatype.QuantizationType.UNIFORM,
                    signed=True,
                    bitwidth=33,
                    shift=[0],
                    offset=[0],
                ),
                shape=[32, 512],  # KERNEL, CH, WIDTH, HEIGHT
                ),
            )
        return [lbir.LayerWrap(fft=fft_config)]

    def is_applicable(self, layers) -> bool:
        return isinstance(layers[0], qkeras.QActivation) and isinstance(
            layers[1], FFTLayer
        )
