import os

import numpy as np
import pytest
import qkeras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

from chisel4ml import chisel4ml_server
from chisel4ml import optimize
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def pytest_sessionstart(session):
    chisel4ml_server.start_server_once()


def pytest_sessionfinish(session, exitstatus):
    server = chisel4ml_server.start_server_once()
    server.stop()


@pytest.fixture(scope="session")
def bnn_qdense_bn_sign_act() -> tf.keras.Model:
    l0 = qkeras.QDense(3, kernel_quantizer=qkeras.binary())
    l1 = tf.keras.layers.BatchNormalization()
    l2 = qkeras.QActivation(qkeras.binary(alpha=1))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=2))
    model.add(l0)
    model.add(l1)
    model.add(l2)
    model.compile(optimizer="adam", loss="squared_hinge", metrics=["accuracy"])

    x_train = [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]  # noqa: F841
    y_train = [0, 1, 1, 0]  # noqa: F841
    # model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=False)
    # model.save_weights(os.path.join(SCRIPT_DIR, 'bnn_qdense_bn_sign_act.h5'))
    model.load_weights(os.path.join(SCRIPT_DIR, "bnn_qdense_bn_sign_act.h5"))
    return model


@pytest.fixture(scope="session")
def bnn_simple_model() -> tf.keras.Model:
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
    return model


@pytest.fixture(scope="session")
def bnn_simple_bweight_model() -> tf.keras.Model:
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
    return model


@pytest.fixture(scope="session")
def bnn_mnist_model() -> tf.keras.Model:
    """Build a dense binarized model in qkeras that uses a single integer layer at the
    start, and batch-norm layers in between the dense layers. The test only checks
    that verilog file was succesfully generated.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28 * 28
    num_classes = 10  # ten unique digits
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    y_train = tf.one_hot(y_train, 10)
    y_train = np.where(y_train < 0.1, -1.0, 1.0)
    y_test = tf.one_hot(y_test, 10)
    y_test = np.where(y_test < 0.1, -1.0, 1.0)

    model = tf.keras.models.Sequential()
    # We don't loose any info here since mnist are 8-bit gray-scale images. We just add
    # this quantization to explicitly encode this for the chisel4ml optimizer.
    model.add(tf.keras.layers.Input(shape=image_vector_size))
    model.add(
        qkeras.QActivation(
            qkeras.quantized_bits(bits=8, integer=8, keep_negative=False)
        )
    )
    model.add(qkeras.QDense(64, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(qkeras.QDense(64, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(qkeras.QDense(64, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(
        qkeras.QDense(
            num_classes, kernel_quantizer=qkeras.binary(alpha=1), activation="binary"
        )
    )

    model.compile(optimizer="adam", loss="squared_hinge", metrics=["accuracy"])

    # model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=False)
    # model.save_weights(os.path.join(SCRIPT_DIR, 'bnn_mnist_model.h5'))
    model.load_weights(os.path.join(SCRIPT_DIR, "bnn_mnist_model.h5"))
    return model


@pytest.fixture(scope="session")
def sint_simple_noscale_model() -> tf.keras.Model:
    w1 = np.array([[1, 2, 3, 4], [-4, -3, -2, -1], [2, -1, 1, 1]])
    b1 = np.array([1, 2, 0, 1])
    w2 = np.array([-1, 4, -3, -1]).reshape(4, 1)
    b2 = np.array([2])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = qkeras.QDense(
        4,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=np.array([1, 1, 1, 1])
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = qkeras.QDense(
        1,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=np.array([1])
        ),
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[4].set_weights([w2, b2])
    return model


@pytest.fixture(scope="session")
def sint_mnist_qdense_relu() -> tf.keras.Model:
    """
    Builds a fully-dense (no conv layers) for mnist. The first layer uses unsigned 8
    bit integers as inputs, but the kernels are all quantized to a 4-bit signed
    integer. The activation functions are all ReLU, except for the output activation
    function, which is a softmax (softmax is ignored in hardware). The model achieves
    around 97% accuracy on the MNIST test dataset.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28 * 28
    num_classes = 10  # ten unique digits
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_train = x_train.astype("float32")
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype("float32")

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    model = tf.keras.models.Sequential()
    # We don't loose any info here since mnist are 8-bit gray-scale images. We just add
    # this quantization to explicitly encode this for the chisel4ml optimizer.
    model.add(tf.keras.layers.Input(shape=image_vector_size))
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=8, integer=8)))

    model.add(
        qkeras.QDense(
            32,
            kernel_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        qkeras.QDense(
            32,
            kernel_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        qkeras.QDense(
            32,
            kernel_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        qkeras.QDense(
            num_classes,
            kernel_quantizer=qkeras.quantized_bits(
                bits=3, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
            activation="softmax",
        )
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=True)
    # model.save_weights(os.path.join(SCRIPT_DIR, 'sint_mnist_qdense_relu.h5'))
    model.load_weights(os.path.join(SCRIPT_DIR, "sint_mnist_qdense_relu.h5"))
    return model


@pytest.fixture(scope="session")
def sint_mnist_qdense_relu_pruned() -> tf.keras.Model:
    """An MNIST model with only fully-connected layers (no conv) that is pruned with TF
    model optimization toolkit.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28 * 28
    num_classes = 10  # ten unique digits
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_train = x_train.astype("float32")
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype("float32")

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.90, begin_step=2000, frequency=100
        )
    }

    kernel_quant_params = {
        "bits": 4,
        "integer": 3,
        "keep_negative": True,
        "alpha": "auto_po2",
    }

    model = tf.keras.models.Sequential()
    # We don't loose any info here since mnist are 8-bit gray-scale images. We just add
    # this quantization to explicitly encode this for the chisel4ml optimizer.
    model.add(tf.keras.layers.Input(shape=image_vector_size))
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=8, integer=8)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                32,
                kernel_quantizer=qkeras.quantized_bits(**kernel_quant_params),
                use_bias=True,
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                32,
                kernel_quantizer=qkeras.quantized_bits(**kernel_quant_params),
                use_bias=True,
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                32,
                kernel_quantizer=qkeras.quantized_bits(**kernel_quant_params),
                use_bias=True,
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                num_classes,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                activation="softmax",
            ),
            **pruning_params,
        )
    )

    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [pruning_callbacks.UpdatePruningStep()]

    model.fit(
        x_train, y_train, batch_size=32, epochs=30, verbose=False, callbacks=callbacks
    )
    # model.save_weights(os.path.join(SCRIPT_DIR, 'sint_mnist_qdense_relu_pruned.h5'))
    # model.load_weights(os.path.join(SCRIPT_DIR, "sint_mnist_qdense_relu_pruned.h5"))
    return model


@pytest.fixture(scope="session")
def sint_simple_model() -> tf.keras.Model:
    w1 = np.array([[1, 2, 3, 4], [-4, -3, -2, -1], [2, -1, 1, 1]])
    b1 = np.array([1, 2, 0, 1])
    w2 = np.array([-1, 4, -3, -1]).reshape(4, 1)
    b2 = np.array([2])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = qkeras.QDense(
        4,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=np.array([0.5, 0.25, 1, 0.25])
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
    x = qkeras.QDense(
        1,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=np.array([0.125])
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[4].set_weights([w2, b2])
    return model


@pytest.fixture(scope="session")
def sint_conv_layer() -> tf.keras.Model:
    # conv2d kernel shape: [height, width, input_channels // groups, filters]
    # The filters are: [1 2
    #                   3 4]
    w1 = np.array([1, 2, 3, 4]).reshape(2, 2, 1, 1)
    b1 = np.array([0])

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
    return model


@pytest.fixture(scope="session")
def sint_conv_layer_2_channels() -> tf.keras.Model:
    # conv2d kernel shape: [height, width, input_channels // groups, filters]
    # The filter channels are: [1 2  and [-4, -3
    #                           3 4]      -2, -1]
    w1 = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1 = np.moveaxis(w1, [1, 2, 3, 0], [0, 1, 3, 2])
    b1 = np.array([0, 0])

    x = x_in = tf.keras.layers.Input(shape=(2, 3, 3))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[2, 2],
        data_format="channels_first",
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].dwconv.set_weights([w1, b1])
    return model


@pytest.fixture(scope="session")
def sint_conv_layer_2_kernels_2_channels() -> tf.keras.Model:
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
    return model


@pytest.fixture(scope="session")
def sint_simple_conv_model() -> tf.keras.Model:
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
    return model


@pytest.fixture(scope="session")
def sint_conv_model_width_noteq_height():
    w1a = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1a = np.moveaxis(w1a, [1, 2, 3, 0], [0, 1, 3, 2])
    w1b = np.array([2, 2, 2, 2, 3, 3, 3, 3]).reshape(2, 2, 2, 1)
    w1b = np.moveaxis(w1b, [1, 2, 3, 0], [0, 1, 3, 2])
    w1 = np.concatenate([w1a, w1b], axis=3)
    b1 = np.array([0, 0, 0, 0])

    # channels, height, width
    x = x_in = tf.keras.layers.Input(shape=(2, 3, 6))
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
    return model


@pytest.fixture(scope="session")
def sint_conv_model_width_noteq_height_2():
    w1a = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1a = np.moveaxis(w1a, [1, 2, 3, 0], [0, 1, 3, 2])
    w1b = np.array([2, 2, 2, 2, 3, 3, 3, 3]).reshape(2, 2, 2, 1)
    w1b = np.moveaxis(w1b, [1, 2, 3, 0], [0, 1, 3, 2])
    w1 = np.concatenate([w1a, w1b], axis=3)
    b1 = np.array([0, 0, 0, 0])

    # channels, height, width
    x = x_in = tf.keras.layers.Input(shape=(2, 6, 3))
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
    return model


@pytest.fixture(scope="session")
def sint_simple_maxpool_model():
    x = x_in = tf.keras.layers.Input(shape=(4, 4, 2))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
    )(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    return model


@pytest.fixture(scope="session")
def sint_simple_conv_maxpool_model():
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

    x = x_in = tf.keras.layers.Input(shape=(5, 5, 2))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=4, keep_negative=False)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[2, 2],
        depth_multiplier=2,
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].dwconv.set_weights([w1, b1])
    return model


@pytest.fixture(scope="session")
def sint_conv_maxpool_model():
    w1a = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1a = np.moveaxis(w1a, [1, 2, 3, 0], [0, 1, 3, 2])
    w1b = np.array([2, 2, 2, 2, 3, 3, 3, 3]).reshape(2, 2, 2, 1)
    w1b = np.moveaxis(w1b, [1, 2, 3, 0], [0, 1, 3, 2])
    w1c = np.array([-1, -2, -3, -4, 3, 2, 1, 0]).reshape(2, 2, 2, 1)
    w1c = np.moveaxis(w1c, [1, 2, 3, 0], [0, 1, 3, 2])
    w1 = np.concatenate([w1a, w1b, w1c], axis=3)
    b1 = np.array([0, 0, 0, 0, 0, 0])

    x = x_in = tf.keras.layers.Input(shape=(5, 5, 2))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=4, keep_negative=False)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[2, 2],
        depth_multiplier=3,
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].dwconv.set_weights([w1, b1])
    return model


@pytest.fixture(scope="session")
def sint_conv_conv_model():
    w1 = np.array([1, 2, 3, 4, -4, -3, -2, -1]).reshape(2, 2, 2, 1)
    w1 = np.moveaxis(w1, [1, 2, 3, 0], [0, 1, 3, 2])
    b1 = np.array([0, 0])

    w2a = np.array([2, 2, 2, 2, 3, 3, 3, 3]).reshape(2, 2, 2, 1)
    w2a = np.moveaxis(w2a, [1, 2, 3, 0], [0, 1, 3, 2])
    w2b = np.array([-1, -2, -3, -4, 3, 2, 1, 0]).reshape(2, 2, 2, 1)
    w2b = np.moveaxis(w2b, [1, 2, 3, 0], [0, 1, 3, 2])
    w2 = np.concatenate([w2a, w2b], axis=3)
    b2 = np.array([0, 0, 0, 0])

    x = x_in = tf.keras.layers.Input(shape=(5, 5, 2))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=4, integer=4, keep_negative=False)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[2, 2],
        depth_multiplier=1,
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[2, 2],
        depth_multiplier=2,
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=7, keep_negative=True)
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].dwconv.set_weights([w1, b1])
    model.layers[4].dwconv.set_weights([w2, b2])
    return model


@pytest.fixture(scope="session")
def sint_digit_model_ds():
    from sklearn import datasets

    digits_ds = datasets.load_digits()
    x = x_in = tf.keras.layers.Input(shape=(8, 8, 1))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=8, keep_negative=False)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[3, 3],
        depth_multiplier=1,
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = tf.keras.layers.Flatten()(x)
    x = qkeras.QDense(
        10,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha="auto_po2"
        ),
        use_bias=False,
    )(x)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=7, keep_negative=True)
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # model.fit(
    #     x=digits_ds.images,
    #     y=digits_ds.target,
    #     batch_size=128,
    #     epochs=10,
    #     validation_split=0.2)
    opt_model = optimize.qkeras_model(model)
    opt_model.summary()
    opt_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # opt_model.save_weights(
    #     os.path.join(SCRIPT_DIR, "sint_digit_model.h5")
    # )  # noqa: E501
    opt_model.load_weights(os.path.join(SCRIPT_DIR, "sint_digit_model.h5"))
    return opt_model, digits_ds


@pytest.fixture(scope="session")
def audio_data():
    train_ds, info = tfds.load(
        "speech_commands",
        split="train",
        with_info=True,
        shuffle_files=False,
        as_supervised=True,
    )
    val_ds = tfds.load(
        "speech_commands", split="validation", shuffle_files=False, as_supervised=True
    )
    test_ds = tfds.load(
        "speech_commands", split="test", shuffle_files=False, as_supervised=True
    )

    label_names = []
    for name in info.features["label"].names:
        print(name, info.features["label"].str2int(name))
        label_names = label_names[:] + [name]

    def get_frames(x):
        npads = (32 * 512) - x.shape[0]
        frames = np.pad(x, (0, npads)).reshape([32, 512])
        frames = np.round(((frames / 2**15)) * 2047 * 0.8)
        return frames.reshape(32, 512)

    def train_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(train_ds),
        )

    def val_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(val_ds),
        )

    def test_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(test_ds),
        )

    train_set = tf.data.Dataset.from_generator(  # noqa: F841
        train_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )

    val_set = tf.data.Dataset.from_generator(  # noqa: F841
        val_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    test_set = tf.data.Dataset.from_generator(  # noqa: F841
        test_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    return [
        train_set,
        val_set,
        test_set,
        label_names,
        len(train_ds),
        len(val_ds),
        len(test_ds),
    ]


@pytest.fixture(scope="session")
def audio_data_preproc():
    train_ds, info = tfds.load(
        "speech_commands",
        split="train",
        with_info=True,
        shuffle_files=False,
        as_supervised=True,
    )
    val_ds = tfds.load(
        "speech_commands", split="validation", shuffle_files=False, as_supervised=True
    )
    test_ds = tfds.load(
        "speech_commands", split="test", shuffle_files=False, as_supervised=True
    )

    label_names = []
    for name in info.features["label"].names:
        print(name, info.features["label"].str2int(name))
        label_names = label_names[:] + [name]

    fft_layer = FFTLayer("hamming")
    lmfe_layer = LMFELayer()

    def preproc(sample):
        return lmfe_layer(fft_layer(np.expand_dims(sample, axis=0)))

    def get_frames(x):
        npads = (32 * 512) - x.shape[0]
        frames = np.pad(x, (0, npads)).reshape([32, 512])
        frames = np.round(((frames / 2**15)) * 2047 * 0.8)
        return preproc(frames.reshape(32, 512))

    def train_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(train_ds),
        )

    def val_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(val_ds),
        )

    def test_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(test_ds),
        )

    train_set = tf.data.Dataset.from_generator(  # noqa: F841
        train_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(None, 32, 20, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )

    val_set = tf.data.Dataset.from_generator(  # noqa: F841
        val_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(None, 32, 20, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    test_set = tf.data.Dataset.from_generator(  # noqa: F841
        test_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(None, 32, 20, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    return [
        train_set,
        val_set,
        test_set,
        label_names,
        len(train_ds),
        len(val_ds),
        len(test_ds),
    ]


@pytest.fixture(scope="session")
def qnn_audio_class_no_preproc_no_bias(audio_data_preproc):
    train_set = audio_data_preproc[0]  # noqa: F841
    val_set = audio_data_preproc[1]  # noqa: F841
    test_set = audio_data_preproc[2]  # noqa: F841
    label_names = audio_data_preproc[3]
    TRAIN_SET_LENGTH = audio_data_preproc[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data_preproc[5]  # noqa: F841
    EPOCHS = 3  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    input_shape = (32, 20, 1)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.90, begin_step=2000, frequency=100
        )
    }

    model = tf.keras.models.Sequential()
    model.add(
        qkeras.QActivation(
            activation=qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1),
            input_shape=input_shape,
        ),
    )
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=1,
            use_bias=False,
            depthwise_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=5, integer=5)))
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=2,
            use_bias=False,
            depthwise_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(FlattenChannelwise())
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                8,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=False,
            ),
            **pruning_params,
        )
    )
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                num_labels,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=False,
            ),
            **pruning_params,
        )
    )
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(bits=8, integer=7, keep_negative=True))
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # model.fit_generator(
    #    train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
    #    steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
    #    validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
    #        EPOCHS
    #    ),  # noqa: E501
    #    validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
    #    epochs=EPOCHS,
    #    verbose=True,
    #    callbacks=[pruning_callbacks.UpdatePruningStep()],
    # )
    # model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    opt_model = optimize.qkeras_model(model)
    opt_model.summary()
    opt_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # opt_model.save_weights(
    #     os.path.join(SCRIPT_DIR, "qnn_audio_class_opt_no_preproc_no_bias.h5")
    # )  # noqa: E501
    opt_model.load_weights(
        os.path.join(SCRIPT_DIR, "qnn_audio_class_opt_no_preproc_no_bias.h5")
    )
    return opt_model


@pytest.fixture(scope="session")
def qnn_audio_class_no_preproc(audio_data_preproc):
    train_set = audio_data_preproc[0]  # noqa: F841
    val_set = audio_data_preproc[1]  # noqa: F841
    test_set = audio_data_preproc[2]  # noqa: F841
    label_names = audio_data_preproc[3]
    TRAIN_SET_LENGTH = audio_data_preproc[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data_preproc[5]  # noqa: F841
    EPOCHS = 3  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    input_shape = (32, 20, 1)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.90, begin_step=2000, frequency=100
        )
    }

    model = tf.keras.models.Sequential()
    model.add(
        qkeras.QActivation(
            activation=qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1),
            input_shape=input_shape,
        ),
    )
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=1,
            use_bias=True,
            bias_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha=1
            ),
            depthwise_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=5, integer=5)))
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=2,
            use_bias=True,
            bias_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha=1
            ),
            depthwise_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(FlattenChannelwise())
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                8,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                bias_quantizer=qkeras.quantized_bits(
                    bits=8, integer=7, keep_negative=True, alpha=1
                ),
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                num_labels,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                bias_quantizer=qkeras.quantized_bits(
                    bits=8, integer=7, keep_negative=True, alpha=1
                ),
            ),
            **pruning_params,
        )
    )
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(bits=8, integer=7, keep_negative=True))
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.build()

    # model.fit_generator(
    #    train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
    #    steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
    #    validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
    #        EPOCHS
    #    ),  # noqa: E501
    #    validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
    #    epochs=EPOCHS,
    #    verbose=True,
    #    callbacks=[pruning_callbacks.UpdatePruningStep()],
    # )
    # model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    opt_model = optimize.qkeras_model(model)
    opt_model.summary()
    opt_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # opt_model.fit_generator(
    #     train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
    #     steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
    #     validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
    #         EPOCHS
    #     ),  # noqa: E501
    #     validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
    #     epochs=EPOCHS,
    #     verbose=True,
    #     callbacks=[pruning_callbacks.UpdatePruningStep()],
    # )
    # opt_model.save_weights(
    #     os.path.join(SCRIPT_DIR, "qnn_audio_class_opt_no_preproc.h5")
    # )  # noqa: E501
    opt_model.load_weights(
        os.path.join(SCRIPT_DIR, "qnn_audio_class_opt_no_preproc.h5")
    )
    return opt_model


@pytest.fixture(scope="session")
def qnn_audio_class(audio_data):
    train_set = audio_data[0]  # noqa: F841
    val_set = audio_data[1]  # noqa: F841
    # test_set = audio_data[2]
    label_names = audio_data[3]
    TRAIN_SET_LENGTH = audio_data[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data[5]  # noqa: F841

    EPOCHS = 3  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    input_shape = (32, 512)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.90, begin_step=2000, frequency=100
        )
    }

    model = tf.keras.models.Sequential()
    model.add(
        qkeras.QActivation(
            qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1),
            input_shape=input_shape,
        )
    )
    model.add(FFTLayer(win_fn="hamming"))
    model.add(LMFELayer())
    model.add(
        qkeras.QActivation(
            activation=qkeras.quantized_bits(8, 7, keep_negative=True, alpha=1),
        ),
    )
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=1,
            use_bias=True,
            bias_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha=1
            ),
            depthwise_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=5, integer=5)))
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=2,
            use_bias=True,
            bias_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha=1
            ),
            depthwise_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(FlattenChannelwise())
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                8,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                bias_quantizer=qkeras.quantized_bits(
                    bits=8, integer=7, keep_negative=True, alpha=1
                ),
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                num_labels,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                bias_quantizer=qkeras.quantized_bits(
                    bits=8, integer=7, keep_negative=True, alpha=1
                ),
            ),
            **pruning_params,
        )
    )
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(bits=8, integer=7, keep_negative=True))
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # model.fit_generator(
    #    train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
    #    steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
    #    validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
    #        EPOCHS
    #    ),  # noqa: E501
    #    validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
    #    epochs=EPOCHS,
    #    verbose=True,
    #    callbacks=[pruning_callbacks.UpdatePruningStep()],
    # )
    # model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    opt_model = optimize.qkeras_model(model)
    opt_model.summary()
    opt_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # opt_model.fit_generator(
    #     train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
    #     steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
    #     validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
    #         EPOCHS
    #     ),  # noqa: E501
    #     validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
    #     epochs=EPOCHS,
    #     verbose=True,
    #     callbacks=[pruning_callbacks.UpdatePruningStep()],
    # )
    # opt_model.save_weights(
    #     os.path.join(SCRIPT_DIR, "qnn_audio_class.h5")
    # )  # noqa: E501
    opt_model.load_weights(os.path.join(SCRIPT_DIR, "qnn_audio_class.h5"))

    # opt_model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    return opt_model
