import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer
import qkeras

import chisel4ml.lbir_python.lbir_pb2 as lbir
from chisel4ml.transforms import register_qkeras_transform

import logging


# TODO: A single quantizer could possibly have several targets.
# I.e. binary-auto and binary-po2
_qkeras2lbir_quantizer_dict = {
        qkeras.quantized_bits: lbir.Quantizer.SYMMETRIC_UNIFORM_PO2,
        qkeras.binary: lbir.Quantizer.BINARY_SIGN
        }


def _log_transform(fn):
    def wrap_trans_fn(layer):
        logging.debug(f"Transforming keras layer {layer.__class__} to a LBIR layer with the "
                      f"{fn.__name__} transform.")
        return fn(layer)
    return wrap_trans_fn


@_log_transform
@register_qkeras_transform(qkeras.QDense)
def transform_qkeras_dense(layer: KerasLayer) -> lbir.Layer:
    if not layer.kernel.dtype == tf.float32:
        raise ValueError("The tensorflow backend should be set to float32!")  # TODO
    if layer.use_bias:
        if not layer.bias.dtype == tf.float32:
            raise ValueError("The tensorflow backend should be set to float32!")  # TODO
    lbir_layer = lbir.Layer()
    lbir_layer.layer_type = lbir._LAYER_LAYERTYPE.values_by_name['DENSE'].number
    lbir_layer.use_bias = layer.use_bias
    lbir_layer.weights.quantizer.type = _qkeras2lbir_quantizer_dict[layer.kernel_quantizer.__class__]
    lbir_layer.weights.quantizer.scale = 1
    lbir_layer.weights.quantizer.offset = 0
    lbir_layer.weights.values.extend(layer.kernel_quantizer_internal(layer.kernel).numpy().tobytes())
    if layer.use_bias:
        lbir_layer.biases.quantizer.type = _qkeras2lbir_quantizer_dict[layer.bias_quantizer.__class__]
        lbir_layer.biases.quantizer.scale = 1
        lbir_layer.biases.quantizer.offset = 0
        lbir_layer.biases.values.extend(layer.bias_quantizer_internal(layer.bias).numpy().tobytes())
    lbir_layer.width = layer.kernel_quantizer_internal(layer.kernel).numpy().shape[0]
    lbir_layer.height = layer.kernel_quantizer_internal(layer.kernel).numpy().shape[1]
    lbir_layer.channels = 0
    return lbir_layer
