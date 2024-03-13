import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="non-trainable")
def case_bnn_simple_model():
    w1 = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    b1 = np.array([1, 2, 0, 1])
    w2 = np.array([-1, 1, -1, -1]).reshape(4, 1)
    b2 = np.array([1])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(qkeras.binary(alpha=1))(x)
    x = qkeras.QDense(4, kernel_quantizer=qkeras.binary(alpha=1), activation="binary")(
        x
    )
    x = qkeras.QDense(1, kernel_quantizer=qkeras.binary(alpha=1), activation="binary")(
        x
    )
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[3].set_weights([w2, b2])
    data = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return model, data
