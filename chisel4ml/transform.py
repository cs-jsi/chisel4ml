from chisel4ml.transforms import qkeras_transform_factory
import chisel4ml.lbir.lbir_pb2 as lbir

import tensorflow as tf


def qkeras_to_lbir(model: tf.keras.Model, name="chisel4ml_model") -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    lbir_model = lbir.Model()
    lbir_model.name = name
    for i, layer in enumerate(model.layers):
        lbir_layer = qkeras_transform_factory(layer)(layer)
        lbir_model.layers.extend([lbir_layer])
    return lbir_model
