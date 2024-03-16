import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="non-trainable")
def case_bnn_simple_bweight_model():
    w1 = np.array([[1, -1], [-1, -1], [1, 1]])
    b1 = np.array([2, 0])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=8, keep_negative=False)
    )(x)
    x = qkeras.QDense(2, kernel_quantizer=qkeras.binary(alpha=1), activation="binary")(
        x
    )
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    data = np.array(
        [
            [36.0, 22.0, 3.0],
            [6.0, 18.0, 5.0],
            [6.0, 22.0, 3.0],
            [255.0, 127.0, 255.0],
            [0.0, 0.0, 0.0],
            [255.0, 255.0, 255.0],
        ]
    )
    return model, data
