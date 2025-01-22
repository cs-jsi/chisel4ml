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
import logging

import onnx
import qonnx.converters
import qonnx.custom_op.registry
import qonnx.util.cleanup

from onnx.onnx_ml_pb2 import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.transforms import AddDummyBiasToConv
from chisel4ml.transforms import AddFFTrealOutputShape
from chisel4ml.transforms import AddInputOrOutputQTensorToReshape
from chisel4ml.transforms import AutoPadToPad
from chisel4ml.transforms import CleanupQTensors
from chisel4ml.transforms import ExtractQuantizedBiasFromConv
from chisel4ml.transforms import InputReluQTensorToQTensor
from chisel4ml.transforms import MergePad
from chisel4ml.transforms import QONNXToLBIR
from chisel4ml.transforms import QuantToQTensor
from chisel4ml.transforms import RemoveFlattenNode
from chisel4ml.transforms import UnquantizedBiasToQTensor
from chisel4ml.transforms import UnquantizedOutputToQTensor
from chisel4ml.transforms import WeightQuantToQTensor

DEFAULT_QONNX_TRANSFORMS = [
    DoubleToSingleFloat(),
    InferDataLayouts(),
    AddFFTrealOutputShape(),
    GemmToMatMul(),
    FoldTransposeIntoQuantInit(),
    InferShapes(),
    RemoveIdentityOps(),
    SortGraph(),
]

QONNX_TO_LBIR_TRANSFORMS = [
    AddDummyBiasToConv(),
    ExtractQuantizedBiasFromConv(),
    ExtractBiasFromConv(),
    WeightQuantToQTensor(),
    QuantToQTensor(),
    AddInputOrOutputQTensorToReshape(),
    UnquantizedBiasToQTensor(),
    UnquantizedOutputToQTensor(),
    InputReluQTensorToQTensor(),
    RemoveFlattenNode(),
    MergePad(),
    AutoPadToPad(),
    QONNXToLBIR(),
    CleanupQTensors(),
]


def qonnx_to_lbir(
    modelwrap: ModelWrapper,
    name="chisel4ml_model",
    custom_trans_list=[],
    cleanup=True,
    debug=False,
) -> lbir.Model:  # type: ignore
    "Applys transformation to a QONNX model, and returns a LBIR model."
    if len(custom_trans_list) == 0:
        transforms = DEFAULT_QONNX_TRANSFORMS
    else:
        transforms = custom_trans_list
    if cleanup:
        modelwrap = qonnx.util.cleanup.cleanup_model(modelwrap)

    for ind, trans in enumerate(transforms + QONNX_TO_LBIR_TRANSFORMS):
        logging.info(f"Running transform {type(trans).__name__}.")
        if debug:
            onnx.save(
                modelwrap.model,
                f"DEBUG_{name}_{ind}_BEFORE_{type(trans).__name__}.onnx",
            )
        modelwrap = modelwrap.transform(trans)
    if debug:
        onnx.save(
            modelwrap.model,
            f"DEBUG_{name}_FINAL_.onnx",
        )

    lbir_model = _uwrap_qonnx_to_lbir(modelwrap, name)
    return lbir_model


def _uwrap_qonnx_to_lbir(
    onnx_model: ModelWrapper, name: str
) -> lbir.Model:  # type: ignore
    if (
        onnx_model.graph.node[0].op_type == "Reshape"
        and onnx_model.graph.node[-1].op_type == "Reshape"
    ):
        # This condition typically arises from QKeras conv models that have different
        # tensor memory layout, hence the reshape ops.
        layers = onnx_model.graph.node[1:-1]
        input_channel_first = True
    else:
        layers = onnx_model.graph.node
        input_channel_first = False
    return lbir.Model(
        name=name,
        layers=[_unwrap_qonnx_layer_to_lbir(lay) for lay in layers],
        input_channel_first=input_channel_first,
    )


def _unwrap_qonnx_layer_to_lbir(layer: NodeProto) -> lbir.LayerWrap:  # type: ignore
    if layer.op_type == "QDense":
        qdense_str = onnx.helper.get_node_attr_value(layer, "qdense")
        return lbir.LayerWrap(dense=lbir.DenseConfig.FromString(qdense_str))
    elif layer.op_type == "QConv":
        qconv_str = onnx.helper.get_node_attr_value(layer, "qconv")
        return lbir.LayerWrap(conv2d=lbir.Conv2DConfig.FromString(qconv_str))
    elif layer.op_type == "MaxPool2D":
        maxpool_str = onnx.helper.get_node_attr_value(layer, "maxpool2d")
        return lbir.LayerWrap(maxpool2d=lbir.MaxPool2DConfig.FromString(maxpool_str))
    elif layer.op_type == "FFTConfig":
        fft_str = onnx.helper.get_node_attr_value(layer, "fft")
        return lbir.LayerWrap(fft=lbir.FFTConfig.FromString(fft_str))
    elif layer.op_type == "LMFEConfig":
        lmfe_str = onnx.helper.get_node_attr_value(layer, "lmfe")
        return lbir.LayerWrap(lmfe=lbir.LMFEConfig.FromString(lmfe_str))
    else:
        raise NotImplementedError
