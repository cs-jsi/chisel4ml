import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case

from chisel4ml.qkeras_extensions import MaxPool2dCF


@case(tags="non-trainable")
def case_sint_simple_maxpool_model():
    x = x_in = tf.keras.layers.Input(shape=(2, 4, 4))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = MaxPool2dCF()(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    x0 = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 5.0, 6.0, 7.0],
        ]
    ).reshape(1, 4, 4)
    x1 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).reshape(1, 4, 4)
    x2 = np.zeros(16).reshape(1, 4, 4)
    x01 = np.concatenate((x0, x1), axis=0)
    x10 = np.concatenate((x1, x0), axis=0)
    x21 = np.concatenate((x2, x1), axis=0)
    x12 = np.concatenate((x1, x2), axis=0)
    data = np.concatenate((x01, x10, x21, x12)).reshape(4, 2, 4, 4)
    return model, data
