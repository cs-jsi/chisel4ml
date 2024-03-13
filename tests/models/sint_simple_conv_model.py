import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="non-trainable")
def case_sint_simple_conv_model():
    # conv2d kernel shape: [height, width, input_channels // groups, filters]
    # The filters is: [1 2
    #                  3 4]
    w1 = np.array([1, 2, 3, 4]).reshape(2, 2, 1, 1)
    b1 = np.array([-2])

    w2 = np.array([-1, 4, -3, -1]).reshape(4, 1)
    b2 = np.array([2])

    x = x_in = tf.keras.layers.Input(shape=(1, 3, 3))  # 3x3 monochrome images
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = qkeras.QConv2D(
        filters=1,
        kernel_size=[2, 2],
        strides=[1, 1],
        data_format="channels_first",
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=np.array([0.25])
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
    x = tf.keras.layers.Flatten()(x)
    x = qkeras.QDense(
        1,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=0.5
        ),
    )(x)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=7, keep_negative=True)
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[5].set_weights([w2, b2])
    x0 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, -1.0]]).reshape(1, 3, 3)
    x1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).reshape(1, 3, 3)
    x2 = np.zeros(9).reshape(1, 3, 3)
    data = np.concatenate((x0, x1, x2)).reshape(3, 1, 3, 3)
    return model, data
