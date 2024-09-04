import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="non-trainable")
def case_sint_conv_conv_model():
    w1 = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1 = np.moveaxis(w1, [1, 2, 3, 0], [0, 1, 3, 2])
    b1 = np.array([0, 0])

    w2a = np.array([2, 2, 2, 2, 3, 3, 3, 3]).reshape(2, 2, 2, 1)
    w2a = np.moveaxis(w2a, [1, 2, 3, 0], [0, 1, 3, 2])
    w2b = np.array([-1, -2, -3, -4, 3, 2, 1, 0]).reshape(2, 2, 2, 1)
    w2b = np.moveaxis(w2b, [1, 2, 3, 0], [0, 1, 3, 2])
    w2 = np.concatenate([w2a, w2b], axis=3)
    b2 = np.array([0, 0, 0, 0])

    x = x_in = tf.keras.layers.Input(shape=(2, 5, 5))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=4, keep_negative=False)
    )(x)
    x = qkeras.QDepthwiseConv2D(
        kernel_size=[2, 2],
        depth_multiplier=1,
        data_format="channels_first",
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = qkeras.QDepthwiseConv2D(
        kernel_size=[2, 2],
        depth_multiplier=2,
        data_format="channels_first",
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=7, keep_negative=True)
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[4].set_weights([w2, b2])
    x0 = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 0, 1, 2, 3],
            [4, 5, 6, 7, 8],
        ]
    ).reshape(1, 5, 5)
    x1 = np.array(
        [
            [9, 10, 11, 12, 13],
            [14, 15, 0, 1, 2],
            [3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12],
            [13, 14, 15, 0, 1],
        ]
    ).reshape(1, 5, 5)
    data = np.concatenate((x0, x1)).reshape(1, 2, 5, 5).astype(np.float32)
    return model, data
