import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="non-trainable")
def case_sint_conv_layer():
    # conv2d kernel shape: [height, width, input_channels // groups, filters]
    # The filters are: [1 2
    #                   3 4]
    w1 = np.array([1, 2, 3, 4]).reshape(2, 2, 1, 1)
    b1 = np.array([1])

    x = x_in = tf.keras.layers.Input(shape=(3, 3, 1))  # 3x3 monochrome images
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = qkeras.QConv2D(
        filters=1,
        kernel_size=[2, 2],
        strides=[1, 1],
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=np.array([1])
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])

    def data_gen():
        np.random.default_rng(42)
        for x in range(20):
            x = np.random.rand(9).reshape(3, 3, 1)  # [0, 1]
            x = (x * 2) - 1  # [-1, 1]
            x = np.round(x * (2**3))
            x = np.clip(x, -(2**3), (2**3) - 1)
            yield x

    data = data_gen()
    return model, data
