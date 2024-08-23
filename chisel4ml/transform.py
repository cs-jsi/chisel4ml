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
import onnx
import qonnx.converters
import qonnx.util.cleanup
import tensorflow as tf
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.remove import RemoveIdentityOps

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.transforms import BiasToQTensor
from chisel4ml.transforms import QONNXToLBIR
from chisel4ml.transforms import QuantToQTensor
from chisel4ml.transforms import WeightQuantToQTensor

DEFAULT_TRANSFORMS = [
    DoubleToSingleFloat(),
    InferDataLayouts(),
    RemoveIdentityOps(),
    SortGraph(),
]


def qkeras_to_lbir(
    qkeras_model: tf.keras.Model,
    name="chisel4ml_model",
    custom_trans_list=[],
    cleanup=True,
) -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    qonnx_proto, _ = qonnx.converters.from_keras(qkeras_model)
    modelwrap = qonnx.core.modelwrapper.ModelWrapper(qonnx_proto)
    if len(custom_trans_list) == 0:
        transforms = DEFAULT_TRANSFORMS
    else:
        transforms = custom_trans_list

    if cleanup:
        modelwrap = qonnx.util.cleanup.cleanup_model(modelwrap)

    for trans in transforms:
        modelwrap = modelwrap.transform(trans)

    modelwrap = modelwrap.transform(WeightQuantToQTensor())
    modelwrap = modelwrap.transform(QuantToQTensor())
    modelwrap = modelwrap.transform(BiasToQTensor())
    modelwrap = modelwrap.transform(QONNXToLBIR())
    onnx.save(modelwrap.model, "test_transform5.onnx")
