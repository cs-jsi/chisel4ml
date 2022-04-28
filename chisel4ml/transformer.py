from chisel4ml.transforms import qkeras_transform_factory
import chisel4ml.lbir_python.lbir_pb2 as lbir

import tensorflow as tf


def transform_model_to_lbir(model: tf.keras.Model) -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    lbir_model = lbir.Model()
    for i, layer in enumerate(model.layers):
        lbir_layer = qkeras_transform_factory(layer)(layer)
        lbir_model.layers.extend([lbir_layer])
    return lbir_model
