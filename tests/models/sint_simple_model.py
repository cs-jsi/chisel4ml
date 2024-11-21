import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="non-trainable")
def case_sint_simple_model():
    w1 = np.array([[1, 2, 3, 3], [-4, -3, -2, -1], [2, -1, 1, 1]])
    b1 = np.array([1, 2, 0, 1])
    w2 = np.array([-1, 3, -3, -1]).reshape(4, 1)
    b2 = np.array([2])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = qkeras.QDense(
        4,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=[0.5, 0.5, 1, 0.5]
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
    x = qkeras.QDense(
        1,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=[0.5]
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[4].set_weights([w2, b2])
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [4.0, 4.0, 4.0],
            [7.0, 7.0, 7.0],
            [6.0, 0.0, 7.0],
            [3.0, 3.0, 3.0],
            [7.0, 0.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 7.0],
        ]
    )
    return model, data
