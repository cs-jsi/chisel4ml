import qkeras
import tensorflow as tf

import chisel4ml
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import MaxPool2dCF
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


def clone_model_from_config(model):
    config = {
        "class_name": model.__class__.__name__,
        "config": model.get_config(),
    }
    custom_objects = {
        "FFTLayer": FFTLayer,
        "LMFELayer": LMFELayer,
        "MaxPool2dCF": MaxPool2dCF,
        "QDepthwiseConv2DPermuted": QDepthwiseConv2DPermuted,
        "FlattenChannelwise": FlattenChannelwise,
    }
    qkeras.utils._add_supported_quantized_objects(custom_objects)
    clone = tf.keras.models.model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def get_submodel(model, num_layers):
    """
    Converts the num_layers argument to an index of this layer in the model.
    The num_layers argument represents the number of active layers (fftlayer,
    lmfelayer, qdense, conv...) used in the model. The returned model has
    num_layers active layers.
    """
    ACTIVE_LAYERS = (
        FFTLayer,
        LMFELayer,
        tf.keras.layers.MaxPooling2D,
        MaxPool2dCF,
        chisel4ml.qkeras_extensions.QDepthwiseConv2DPermuted,
        qkeras.QDense,
        qkeras.QDepthwiseConv2D,
        qkeras.Conv2D,
    )
    if not isinstance(num_layers, int) and num_layers > 0:
        raise ValueError(
            f"Invalid argument num_layers:{num_layers}. Should be a positive integer."
        )
    clone_model = clone_model_from_config(model)
    active_layers_index = []
    for active_ind, layer in enumerate(clone_model.layers):
        if isinstance(layer, ACTIVE_LAYERS):
            active_layers_index.append(active_ind)
    if num_layers > len(active_layers_index):
        raise ValueError(
            f"""Model has only {len(active_layers_index)} active layers. The requested
             num_layers is: {num_layers}"""
        )
    index = active_layers_index[num_layers - 1]
    if len(clone_model.layers) > index + 1 and isinstance(
        clone_model.layers[index + 1], qkeras.QActivation
    ):
        index = index + 1
    return tf.keras.models.Model(clone_model.input, clone_model.layers[index].output)
