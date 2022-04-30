from tensorflow.keras.layers import Layer as KerasLayer
import qkeras

import chisel4ml.lbir_python.lbir_pb2 as lbir
from chisel4ml.transforms import register_qkeras_transform

import logging
from collections import defaultdict

_qkeras2lbir_quantizer_dict = defaultdict(lambda: lbir.Datatype.UNIFORM)  # type: ignore
_qkeras2lbir_quantizer_dict[qkeras.quantized_bits] = lbir.Datatype.UNIFORM
_qkeras2lbir_quantizer_dict[qkeras.binary] = lbir.Datatype.BINARY


def _log_transform(fn):
    def wrap_trans_fn(layer):
        logging.debug(f"Transforming keras layer {layer.__class__} to a LBIR layer with the "
                      f"{fn.__name__} transform.")
        return fn(layer)
    return wrap_trans_fn


def _qkeras_base_transform(keras_layer: KerasLayer) -> lbir.Layer:
    """ The base tranform transform transforms the parts of the layer that are common to all layers. """
    lbir_layer = lbir.Layer()

    lbir_layer.input.dtype.quantization = _qkeras2lbir_quantizer_dict[keras_layer.kernel_quantizer.__class__]
    lbir_layer.input.dtype.scale = 1
    lbir_layer.input.dtype.offset = 0
    lbir_layer.input.shape[:] = keras_layer.input_shape[1:]  # We throw away the batch dimension

    lbir_layer.activation.fn = lbir.Activation.BINARY_SIGN
    lbir_layer.activation.bitwidth = 1

    lbir_layer.out_shape[:] = keras_layer.output_shape[1:]

    lbir_layer.weights.dtype.quantization = _qkeras2lbir_quantizer_dict[keras_layer.kernel_quantizer.__class__]
    lbir_layer.weights.dtype.scale = 1
    lbir_layer.weights.dtype.offset = 0
    lbir_layer.weights.values[:] = keras_layer.kernel_quantizer_internal(keras_layer.kernel).numpy().tobytes()

    lbir_layer.use_bias = keras_layer.use_bias
    if keras_layer.use_bias:
        lbir_layer.biases.dtype.quantization = _qkeras2lbir_quantizer_dict[keras_layer.bias_quantizer.__class__]
        lbir_layer.biases.dtype.scale = 1
        lbir_layer.biases.dtype.offset = 0
        if keras_layer.bias_quantizer_internal is not None:
            lbir_layer.biases.values[:] = keras_layer.bias_quantizer_internal(keras_layer.bias).numpy().tobytes()
        else:
            lbir_layer.biases.values[:] = keras_layer.bias.numpy().tobytes()
    return lbir_layer


@_log_transform
@register_qkeras_transform(qkeras.QDense)
def transform_qkeras_dense(keras_layer: KerasLayer) -> lbir.Layer:
    """ Transforms the qkeras dense layer to an equivalent lbir layer. """
    lbir_layer = _qkeras_base_transform(keras_layer)
    lbir_layer.ltype = lbir.Layer.DENSE

    return lbir_layer
