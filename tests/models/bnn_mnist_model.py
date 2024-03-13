import numpy as np
import pytest
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="trainable")
@pytest.mark.skip(reason="to expensive to run")
def case_bnn_mnist_model(mnist_data):
    """Build a dense binarized model in qkeras that uses a single integer layer at the
    start, and batch-norm layers in between the dense layers. The test only checks
    that verilog file was succesfully generated.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist_data

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
    data = {"X_train": x_train, "y_train": y_train, "X_test": x_test, "y_test": y_test}
    training_info = {"epochs": 10, "batch_size": 64, "callbacks": None}
    return (model, data, training_info)
