import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case

from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


@case(tags="non-trainable")
def case_sint_conv_layer_2_kernels_2_channels():
    # conv2d kernel shape: [height, width, input_channels // groups, filters]
    # The filters are: [1 2  and [-4, -3 for filter 0
    #                   3 4]      -2, -1]
    # and [2 2    [3 3
    #      2 2]    3 3]  for filter 1
    w1a = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1a = np.moveaxis(w1a, [1, 2, 3, 0], [0, 1, 3, 2])
    w1b = np.array([2, 2, 2, 2, 3, 3, 3, 3]).reshape(2, 2, 2, 1)
    w1b = np.moveaxis(w1b, [1, 2, 3, 0], [0, 1, 3, 2])
    w1 = np.concatenate([w1a, w1b], axis=3)
    b1 = np.array([0, 0, 0, 0])

    x = x_in = tf.keras.layers.Input(shape=(2, 3, 3))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[2, 2],
        depth_multiplier=2,
        data_format="channels_first",
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].dwconv.set_weights([w1, b1])
    x0 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, -1.0]]).reshape(1, 3, 3)
    x1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).reshape(1, 3, 3)
    x2 = np.zeros(9).reshape(1, 3, 3)
    x01 = np.concatenate((x0, x1))
    x10 = np.concatenate((x1, x0))
    x21 = np.concatenate((x2, x1))
    x12 = np.concatenate((x1, x2))
    data = np.concatenate((x01, x10, x21, x12)).reshape(4, 2, 3, 3)
    return model, data
