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
import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.lbir.datatype_pb2 import Datatype
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from chisel4ml.transforms import register_qkeras_transform
from chisel4ml.transforms.qkeras_transforms import QKerasTransform


@register_qkeras_transform
class QKerasAudioPreprocess(QKerasTransform):
    """
    Transforms an audio preprocess layer into the LBIR representation.
    """

    num_layers = 1
    order = 3

    def _call_impl(self, layers):
        lmfe_config = layers[0].cfg
        lmfe_config.input.CopyFrom(
            QTensor(
                dtype=Datatype(
                    quantization=Datatype.QuantizationType.UNIFORM,
                    signed=True,
                    bitwidth=33,
                    shift=[0],
                    offset=[0],
                ),
                shape=[layers[0].cfg.num_frames, layers[0].cfg.fft_size],
            )
        )
        lmfe_config.output.CopyFrom(
            QTensor(
                dtype=Datatype(
                    quantization=Datatype.QuantizationType.UNIFORM,
                    signed=True,
                    bitwidth=8,
                    shift=[0],
                    offset=[0],
                ),
                shape=[layers[0].cfg.num_frames, layers[0].cfg.num_mels],
            )
        )
        return [lbir.LayerWrap(lmfe=lmfe_config)]

    def is_applicable(self, layers) -> bool:
        return isinstance(layers[0], LMFELayer)
