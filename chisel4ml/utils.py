import qkeras
import tensorflow as tf

from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


def clone_model_from_config(model):
    config = {
        "class_name": model.__class__.__name__,
        "config": model.get_config(),
    }
    custom_objects = {
        "FFTLayer": FFTLayer,
        "LMFELayer": LMFELayer,
        "QDepthwiseConv2DPermuted": QDepthwiseConv2DPermuted,
        "FlattenChannelwise": FlattenChannelwise,
    }
    qkeras.utils._add_supported_quantized_objects(custom_objects)
    clone = tf.keras.models.model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
