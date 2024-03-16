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
from typing import Iterable
from typing import List

import numpy as np
import qkeras
import tensorflow as tf
from qkeras import QConv2D
from qkeras import QDense
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.layers import MaxPooling2D

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.lbir.datatype_pb2 import Datatype as LBIRDatatype
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.qkeras_extensions import MaxPool2dCF
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted

log = logging.getLogger(__name__)


def _qkeras_base_transform_no_inp(keras_layer: KerasLayer):
    """The base tranform transform transforms the parts of the layer that are common
    to all layers.
    """
    if isinstance(keras_layer, QDense):
        return lbir.LayerWrap(
            dense=lbir.DenseConfig(
                thresh=_layer_to_thresh_tensor(keras_layer),
                kernel=_layer_to_weight_tensor(keras_layer),
                output=_layer_to_output_tensor(keras_layer),
                activation=_qact_to_act(keras_layer.activation),
                rounding_mode=_qact_to_rounding_mode(keras_layer.activation),
            )
        )
    elif isinstance(keras_layer, QConv2D):
        return lbir.LayerWrap(
            conv2d=lbir.Conv2DConfig(
                thresh=_layer_to_thresh_tensor(keras_layer),
                kernel=_layer_to_weight_tensor(keras_layer),
                output=_layer_to_output_tensor(keras_layer),
                activation=_qact_to_act(keras_layer.activation),
                rounding_mode=_qact_to_rounding_mode(keras_layer.activation),
            )
        )
    elif isinstance(keras_layer, QDepthwiseConv2DPermuted):
        return lbir.LayerWrap(
            conv2d=lbir.Conv2DConfig(
                thresh=_layer_to_thresh_tensor(keras_layer),
                kernel=_depthwise_layer_to_weight_tensor(keras_layer),
                output=_layer_to_output_tensor(keras_layer),
                activation=_qact_to_act(keras_layer.activation),
                depthwise=True,
                rounding_mode=_qact_to_rounding_mode(keras_layer.activation),
            )
        )
    elif isinstance(keras_layer, (MaxPooling2D, MaxPool2dCF)):
        return lbir.LayerWrap(
            maxpool2d=lbir.MaxPool2DConfig(
                output=_layer_to_output_tensor(keras_layer),
            )
        )
    else:
        raise NotImplementedError(
            f"Layer of type {type(keras_layer)} is not yet supported by chisel4ml."
        )


def _layer_to_thresh_tensor(keras_layer: KerasLayer) -> QTensor:
    if keras_layer.bias_quantizer_internal is None:
        keras_layer.bias_quantizer_internal = qkeras.quantized_bits(
            bits=16, integer=15, keep_negative=True, alpha=1
        )
        log.warning(
            "The bias tensor was left unquantized. Adding 16-bit signed integer"
            " quantization."
        )
    else:
        if keras_layer.bias_quantizer_internal.scale != 1:
            raise ValueError(
                "The bias must be quantized with a scale factor of 1. This can be done"
                " by setting the factor alpha to constant 1."
            )

    if isinstance(keras_layer, QDense):
        num_biases = keras_layer.output_shape[1]
        num_shifts = 1
    elif isinstance(keras_layer, (QConv2D, QDepthwiseConv2DPermuted)):
        num_biases = keras_layer.output.shape[-1]
        num_shifts = keras_layer.output.shape[-1]
    else:
        raise Exception(f"Invalid layer type: {type(keras_layer)}")
    bias_values = np.zeros(num_biases)
    if keras_layer.use_bias:
        bias_values = get_integer_values(
            keras_layer.bias, keras_layer.bias_quantizer_internal
        ).numpy()
    thresh_values = (bias_values * (-1.0)).flatten().tolist()
    return QTensor(
        dtype=LBIRDatatype(
            quantization=_quantizer_to_qtype(keras_layer.bias_quantizer_internal),
            signed=True,  # Some way to limit biases to only positive/only negative?
            bitwidth=_quantizer_to_bitwidth(keras_layer.bias_quantizer_internal),
            shift=np.zeros(num_shifts).astype(np.int32),
            offset=[0],
        ),
        shape=[keras_layer.output_shape[1]],
        values=thresh_values,
    )


def _depthwise_layer_to_weight_tensor(keras_layer: KerasLayer) -> QTensor:
    _ = keras_layer.depthwise_quantizer_internal(keras_layer.depthwise_kernel)
    kernel_vals = get_integer_values(
        keras_layer.depthwise_kernel, keras_layer.depthwise_quantizer_internal
    ).numpy()
    kernel_vals = np.moveaxis(kernel_vals, -1, 0)  # move the kernel dimension
    kernel_vals = np.moveaxis(kernel_vals, -1, 1)  # move the channel dimension
    return QTensor(
        dtype=LBIRDatatype(
            quantization=_quantizer_to_qtype(keras_layer.depthwise_quantizer_internal),
            signed=True,  # can this be unsigned in some case?
            bitwidth=_quantizer_to_bitwidth(keras_layer.depthwise_quantizer_internal),
            shift=_quantizer_to_shift(
                keras_layer.depthwise_quantizer_internal, keras_layer.depthwise_kernel
            ),
            offset=[0],
        ),
        shape=_layer_to_shape(keras_layer),
        values=kernel_vals.flatten().tolist(),
    )


def _layer_to_weight_tensor(keras_layer: KerasLayer) -> QTensor:
    # We run this so that scale wont be a place holder tensor
    _ = keras_layer.kernel_quantizer_internal(keras_layer.kernel)
    kernel_vals = get_integer_values(
        keras_layer.kernel, keras_layer.kernel_quantizer_internal
    ).numpy()
    if isinstance(keras_layer, qkeras.QConv2D):
        # LBIR Layout is NCHW!
        kernel_vals = np.moveaxis(kernel_vals, -1, 0)
        kernel_vals = np.moveaxis(kernel_vals, -1, 1)
    return QTensor(
        dtype=LBIRDatatype(
            quantization=_quantizer_to_qtype(keras_layer.kernel_quantizer_internal),
            signed=True,  # can this be unsigned in some case?
            bitwidth=_quantizer_to_bitwidth(keras_layer.kernel_quantizer_internal),
            shift=_quantizer_to_shift(
                keras_layer.kernel_quantizer_internal, keras_layer.kernel
            ),
            offset=[0],
        ),
        shape=_layer_to_shape(keras_layer),
        values=kernel_vals.flatten().tolist(),
    )


def _layer_to_shape(keras_layer: KerasLayer):
    if isinstance(keras_layer, QDense):
        return keras_layer.kernel.shape.as_list()[::-1]
    elif isinstance(keras_layer, QConv2D):
        return list(np.moveaxis(keras_layer.kernel, [0, 1, 2, 3], [3, 2, 1, 0]).shape)
    elif isinstance(keras_layer, QDepthwiseConv2DPermuted):
        return list(
            np.moveaxis(keras_layer.depthwise_kernel, [0, 1, 2, 3], [3, 2, 1, 0]).shape
        )
    else:
        raise ValueError(
            f"Invalid layer of type: {keras_layer.__class__}. Only the QDense and"
            " QConv2D active layers may be used with chisel4ml."
        )


def _layer_to_output_tensor(keras_layer: KerasLayer) -> QTensor:
    tf_shape = keras_layer.get_output_shape_at(0)[1:]
    if isinstance(keras_layer, qkeras.QDense):
        lbir_shape = tf_shape
    elif (
        isinstance(keras_layer, MaxPool2dCF)
        and keras_layer.input_format == "channels_first"
    ):
        lbir_shape = tf_shape
    elif keras_layer.data_format == "channels_first":
        lbir_shape = tf_shape
    else:
        lbir_shape = (tf_shape[2],) + tf_shape[0:2]
    return QTensor(
        dtype=LBIRDatatype(
            quantization=_qact_to_qtype(keras_layer.activation),
            signed=_qact_to_sign(keras_layer.activation),
            bitwidth=_qact_to_bitwidth(keras_layer.activation),
            shift=_qact_to_shift(keras_layer.activation, [1]),
            offset=[0],
        ),
        shape=lbir_shape,
    )


def _qact_to_shift(activation, output_shape):
    num_outputs = 1
    for x in output_shape:
        num_outputs = num_outputs * x
    if isinstance(activation, qkeras.quantized_relu):
        return [activation.bits - activation.integer] * num_outputs
    elif isinstance(activation, qkeras.quantized_bits):
        return [
            activation.bits - (activation.integer + activation.keep_negative)
        ] * num_outputs
    elif isinstance(activation, qkeras.binary):
        return [0] * num_outputs
    elif isinstance(activation, str):
        if activation == "linear" or activation == "softmax":
            return [0] * num_outputs
    elif callable(activation):
        if activation.__name__ == "linear" or activation.__name__ == "softmax":
            return [0] * num_outputs
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}. Only quantized_relu,"
            " binary, linear and softmax are supported currently."
        )


def _qact_to_qtype(activation) -> LBIRDatatype.QuantizationType:
    if isinstance(activation, (qkeras.quantized_relu, qkeras.quantized_bits)):
        return LBIRDatatype.QuantizationType.UNIFORM
    elif isinstance(activation, qkeras.binary):
        return LBIRDatatype.QuantizationType.BINARY
    elif isinstance(activation, str):
        if activation == "linear" or activation == "softmax":
            return LBIRDatatype.QuantizationType.UNIFORM
    elif callable(activation):
        if activation.__name__ == "linear" or activation.__name__ == "softmax":
            return LBIRDatatype.QuantizationType.UNIFORM
    else:
        raise ValueError(
            f"Unsupported activation function: {type(activation)}. Only quantized_relu,"
            " binary, linear and softmax are supported currently."
        )


def _qact_to_act(activation) -> lbir.Activation:
    if isinstance(activation, qkeras.quantized_relu):
        return lbir.Activation.RELU
    elif isinstance(activation, qkeras.binary):
        return lbir.Activation.BINARY_SIGN
    elif isinstance(activation, qkeras.quantized_bits):
        return lbir.Activation.NO_ACTIVATION
    elif isinstance(activation, str):
        if activation == "linear" or activation == "softmax":
            return lbir.Activation.NO_ACTIVATION
    elif callable(activation):
        if activation.__name__ == "linear" or activation.__name__ == "softmax":
            return lbir.Activation.NO_ACTIVATION
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}. Only quantized_relu,"
            " binary, linear and softmax are supported currently."
        )


def _qact_to_sign(activation) -> bool:
    if isinstance(activation, qkeras.quantized_relu):
        return False
    elif isinstance(activation, qkeras.binary):
        return True
    elif isinstance(activation, qkeras.quantized_bits):
        return activation.keep_negative
    elif isinstance(activation, str):
        if activation == "linear" or activation == "softmax":
            return True
    elif callable(activation):
        if activation.__name__ == "linear" or activation.__name__ == "softmax":
            return True
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}. Only quantized_relu,"
            " binary, linear and softmax are supported currently."
        )


def _qact_to_bitwidth(activation) -> int:
    if isinstance(activation, (qkeras.quantized_relu, qkeras.quantized_bits)):
        return activation.bits
    elif isinstance(activation, qkeras.binary):
        return 1
    elif isinstance(activation, str):
        if activation == "softmax" or activation == "linear":
            return 8  # TODO: change to the minimum required bitwidth?
    elif callable(activation):
        if activation.__name__ == "softmax" or activation.__name__ == "linear":
            return 8  # TODO: change to the minimum required bitwidth?
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}. Only quantized_relu,"
            " binary, linear and softmax are supported currently."
        )


def _quantizer_to_qtype(
    quantizer: qkeras.BaseQuantizer,
) -> LBIRDatatype.QuantizationType:
    if isinstance(quantizer, qkeras.quantized_bits):
        return LBIRDatatype.QuantizationType.UNIFORM
    elif isinstance(quantizer, qkeras.binary):
        return LBIRDatatype.QuantizationType.BINARY
    else:
        raise ValueError(
            f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary"
            " quantizers may be used."
        )


def _quantizer_to_bitwidth(quantizer: qkeras.BaseQuantizer) -> int:
    if isinstance(quantizer, qkeras.quantized_bits):
        return quantizer.bits
    elif isinstance(quantizer, qkeras.binary):
        return 1
    else:
        raise ValueError(
            f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary"
            " quantizers may be used."
        )


def _quantizer_to_shift(quantizer: qkeras.BaseQuantizer, tensor) -> List[int]:
    scale = get_scale(quantizer, tensor)
    adjusted_scale = []
    if isinstance(quantizer, qkeras.quantized_bits):
        decimal_bits = quantizer.bits - (quantizer.integer + quantizer.keep_negative)
        adjusted_scale = np.array(scale / (2**decimal_bits))
    elif isinstance(quantizer, qkeras.binary):
        adjusted_scale = scale
    else:
        raise ValueError(
            f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary"
            " quantizers may be used."
        )
    adjusted_scale = np.log2(adjusted_scale)
    adjusted_scale_list = _flatten([adjusted_scale.tolist()])
    adjusted_scale_list = list(map(int, adjusted_scale_list))
    if len(adjusted_scale_list) == 1:
        adjusted_scale_list = adjusted_scale_list * tensor.shape[-1]
    return adjusted_scale_list


def get_scale(quantizer: qkeras.BaseQuantizer, tensor) -> np.ndarray:
    if isinstance(quantizer, qkeras.quantized_bits):
        # We run this so that scale wont be a place holder tensor
        _ = quantizer(tensor)
        return np.array(quantizer.scale)
    elif isinstance(quantizer, qkeras.binary):
        if quantizer.scale != 1:
            raise ValueError(
                "The binary quantizer alpha/scale factor must be set to 1. Other"
                " values are currently not supported."
            )
        return np.array([1])
    else:
        raise ValueError(
            f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary"
            " quantizers may be used."
        )


def get_integer_values(
    values: np.ndarray, quantizer: qkeras.BaseQuantizer
) -> np.ndarray:
    _ = quantizer(values)
    return quantizer(values) / get_scale(quantizer, values)


def get_input_quantization(model):
    if isinstance(model.layers[0], qkeras.qlayers.QActivation):
        return model.layers[0].activation
    elif isinstance(model.layers[0], tf.keras.layers.InputLayer) and isinstance(
        model.layers[1], qkeras.qlayers.QActivation
    ):
        return model.layers[1].activation
    else:
        raise ValueError(
            "model.layers[0] should be a qkeras activation function. This means it"
            " should be subclass of qkeras.qlayer.QActivation. Instead it is"
            f" {type(model.layers[0])}."
        )


def _flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in _flatten(x):
                yield sub_x
        else:
            yield x


def _qact_to_rounding_mode(activation):
    if isinstance(activation, qkeras.quantized_relu):
        return lbir.RoundingMode.ROUND_HALF_TO_EVEN
    elif isinstance(activation, qkeras.binary):
        return lbir.RoundingMode.ROUND_NONE
    elif isinstance(activation, qkeras.quantized_bits):
        if activation.alpha == "alpha_po2":
            return lbir.RoundingMode.ROUND_UP
        elif activation.alpha is None:
            return lbir.RoundingMode.ROUND_HALF_TO_EVEN
        else:
            raise NotImplementedError
    elif isinstance(activation, str):
        if activation == "linear" or activation == "softmax":
            return lbir.RoundingMode.ROUND_NONE
    elif callable(activation):
        if activation.__name__ == "linear" or activation.__name__ == "softmax":
            return lbir.RoundingMode.ROUND_NONE
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}. Only quantized_relu,"
            " binary, linear and softmax are supported currently."
        )
