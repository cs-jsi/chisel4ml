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

import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import linear
import qkeras
import numpy as np
import math
from typing import Iterable

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.transforms import register_qkeras_transform

from typing import Sequence
import inspect


def _qkeras_base_transform_no_inp(keras_layer: KerasLayer) -> lbir.Layer:
    """ The base tranform transform transforms the parts of the layer that are common to all layers. """
    return lbir.Layer(
                ltype = _layer_to_ltype(keras_layer),
                thresh = _layer_to_thresh_tensor(keras_layer),
                weights = _layer_to_weight_tensor(keras_layer),
                output = _layer_to_output_tensor(keras_layer),
                activation = _qact_to_act(keras_layer.activation)
        )


def _layer_to_ltype(keras_layer: KerasLayer) -> lbir.Layer.Type:
    if isinstance(keras_layer, qkeras.qlayers.QDense):
        return lbir.Layer.Type.DENSE
    elif isinstance(keras_layer, qkeras.qconvolutional.QConv2D):
        return lbir.Layer.Type.CONV2D
    else:
        raise ValueError(f"Invalid layer of type: {keras_layer.__class}. Only the QDense and QConv2D active "
                           "layers may be used with chisel4ml.")


def _layer_to_thresh_tensor(keras_layer: KerasLayer) -> lbir.QTensor:
    assert keras_layer.use_bias, ("All layers should use bias. Regardles of the starting settings, after optimization"
                                  " the use_bias settings get switched to true (to enable folding).")
    if keras_layer.bias_quantizer_internal is None:
        keras_layer.bias_quantizer_internal = qkeras.quantized_bits(bits=32, integer=31, keep_negative=True, alpha=1)

    bias_values = get_integer_values(keras_layer.bias, keras_layer.bias_quantizer_internal).numpy()
    thresh_values = (bias_values * (-1.)).flatten().tolist()
    return lbir.QTensor(
        dtype = lbir.Datatype(
            quantization = _quantizer_to_qtype(keras_layer.bias_quantizer_internal),
            signed = True,  # Is there some way to limit biases to be only positive or only negative?
            bitwidth = _quantizer_to_bitwidth(keras_layer.bias_quantizer_internal),
            shift = _quantizer_to_shift(keras_layer.bias_quantizer_internal, keras_layer.bias),
            offset = [0],
        ),
        shape = keras_layer.bias.shape.as_list(),
        values = thresh_values
    )


def _layer_to_weight_tensor(keras_layer: KerasLayer) -> lbir.QTensor:
    # We run this so that scale wont be a place holder tensor
    _ = keras_layer.kernel_quantizer_internal(keras_layer.kernel) 
    return lbir.QTensor(
        dtype = lbir.Datatype(
            quantization = _quantizer_to_qtype(keras_layer.kernel_quantizer_internal),
            signed = True,  # can this be unsigned in some case?
            bitwidth = _quantizer_to_bitwidth(keras_layer.kernel_quantizer_internal),
            shift = _quantizer_to_shift(keras_layer.kernel_quantizer_internal, keras_layer.kernel),
            offset = [0],
        ),
        shape = keras_layer.kernel.shape.as_list(),
        values = get_integer_values(keras_layer.kernel, keras_layer.kernel_quantizer_internal).numpy().flatten().tolist()
    )


def _layer_to_output_tensor(keras_layer: KerasLayer) -> lbir.QTensor:
    return lbir.QTensor(
        dtype = lbir.Datatype(
            quantization = _qact_to_qtype(keras_layer.activation),
            signed = _qact_to_sign(keras_layer.activation),
            bitwidth = _qact_to_bitwidth(keras_layer.activation),
            shift = _qact_to_shift(keras_layer.activation, keras_layer.get_output_shape_at(0)[1:]),
            offset = [0]
        ),
        shape = keras_layer.get_output_shape_at(0)[1:]
    )


def _qact_to_shift(activation, output_shape):
    num_outputs = 1
    for x in output_shape:
        num_outputs = num_outputs * x
    if isinstance(activation, qkeras.quantized_relu):
        return [activation.bits - activation.integer] * num_outputs
    elif isinstance(activation, qkeras.quantized_bits):
        return [activation.bits - (activation.integer + activation.keep_negative)] * num_outputs
    elif isinstance(activation, qkeras.binary):
        return [0] * num_outputs
    elif (activation.__name__ == 'linear' or
          activation.__name__ == 'softmax'):
        return [0] * num_outputs
    else:
        raise ValueError(f"Unsupported activation function: {activation}. Only quantized_relu, binary, linear and "
                         f"softmax are supported currently.")


def _qact_to_qtype(activation) -> lbir.Datatype.QuantizationType:
    if isinstance(activation, (qkeras.quantized_relu, qkeras.quantized_bits)):
        return lbir.Datatype.QuantizationType.UNIFORM
    elif isinstance(activation, qkeras.binary):
        return lbir.Datatype.QuantizationType.BINARY
    elif (activation.__name__ == 'linear' or
          activation.__name__ == 'softmax'):
        return lbir.Datatype.QuantizationType.UNIFORM
    else:
        raise ValueError(f"Unsupported activation function: {type(activation)}. Only quantized_relu, binary, linear and "
                         f"softmax are supported currently.")


def _qact_to_act(activation) -> lbir.Layer.Activation:
    if isinstance(activation, qkeras.quantized_relu):
        return lbir.Layer.Activation.RELU
    elif isinstance(activation, qkeras.binary):
        return lbir.Layer.Activation.BINARY_SIGN
    elif (activation.__name__ == 'linear' or
          activation.__name__ == 'softmax'):
        return lbir.Layer.Activation.NO_ACTIVATION
    else:
        raise ValueError(f"Unsupported activation function: {activation}. Only quantized_relu, binary, linear and "
                         f"softmax are supported currently.")


def _qact_to_sign(activation) -> bool:
    if isinstance(activation, qkeras.quantized_relu):
        return False
    elif isinstance(activation, (qkeras.binary, qkeras.quantized_bits)):
        return True
    elif (activation.__name__ == 'linear' or
          activation.__name__ == 'softmax'):
        return True
    else:
        raise ValueError(f"Unsupported activation function: {activation}. Only quantized_relu, binary, linear and "
                         f"softmax are supported currently.")


def _qact_to_bitwidth(activation) -> int:
    if isinstance(activation, (qkeras.quantized_relu, qkeras.quantized_bits)):
        return activation.bits
    elif isinstance(activation, qkeras.binary):
        return 1
    elif (activation.__name__ == 'softmax' or
          activation.__name__ == 'linear'):
        return 32  # TODO: change to the minimum required bitwidth 
    else:
        raise ValueError(f"Unsupported activation function: {activation}. Only quantized_relu, binary, linear and "
                         f"softmax are supported currently.")


def _quantizer_to_qtype(quantizer: qkeras.BaseQuantizer) -> lbir.Datatype.QuantizationType:
    if isinstance(quantizer, qkeras.quantized_bits):
        return lbir.Datatype.QuantizationType.UNIFORM
    elif isinstance(quantizer, qkeras.binary):
        return lbir.Datatype.QuantizationType.BINARY
    else:
        raise ValueError(f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary quantizers may be used.")


def _quantizer_to_bitwidth(quantizer: qkeras.BaseQuantizer) -> int:
    if isinstance(quantizer, qkeras.quantized_bits):
        return quantizer.bits
    elif isinstance(quantizer, qkeras.binary):
        return 1
    else:
        raise ValueError(f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary quantizers may be used.")


def _quantizer_to_shift(quantizer: qkeras.BaseQuantizer, tensor) -> list[int]:
    scale = get_scale(quantizer, tensor)
    adjusted_scale = []
    if isinstance(quantizer, qkeras.quantized_bits):
        adjusted_scale = np.array(scale / 2**(quantizer.bits - (quantizer.integer + quantizer.keep_negative)))
    elif isinstance(quantizer, qkeras.binary):
        adjusted_scale = scale
    else:
        raise ValueError(f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary quantizers may be used.")
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
            raise ValueError(f"The binary quantizer alpha/scale factor must be set to 1. Other values are currently "
                             f"not supported.")
        return np.array([1])
    else:
        raise ValueError(f"Unsupported quantizer: {quantizer}. Only quantized_bits and binary quantizers may be used.")


def get_integer_values(values: np.ndarray, quantizer: qkeras.BaseQuantizer) -> np.ndarray:
    qvals = quantizer(values)
    if isinstance(quantizer, qkeras.quantized_bits):
        return (quantizer(values) / 
                get_scale(quantizer, values) * 2**(quantizer.bits - (quantizer.integer + quantizer.keep_negative)))
    else:
        return quantizer(values) / get_scale(quantizer, values)


def get_input_quantization(model):
    assert isinstance(model.layers[0], tf.keras.layers.InputLayer)
    if isinstance(model.layers[1], qkeras.qlayers.QActivation):
        return model.layers[1].activation
    else:
        raise ValueError(f"model.layers[1] should be a qkeras activation function. This means it should be subclass of "
                         f"qkeras.qlayer.QActivation. Instead it is {type(model.layers[1])}.")


def _flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in _flatten(x):
                yield sub_x
        else:
            yield x
