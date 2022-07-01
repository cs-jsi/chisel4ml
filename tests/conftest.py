import pytest
import tensorflow as tf
import numpy as np
import qkeras
from qkeras.utils import load_qmodel, model_save_quantized_weights 
import os
from tensorflow.keras.datasets import mnist


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope='session')
def bnn_qdense_bn_sign_act() -> tf.keras.Model:
    l0 = qkeras.QDense(3, kernel_quantizer=qkeras.binary())
    l1 = tf.keras.layers.BatchNormalization()
    l2 = qkeras.QActivation(qkeras.binary(alpha=1))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=2))
    model.add(l0)
    model.add(l1)
    model.add(l2)
    model.compile(optimizer="adam",
                  loss='squared_hinge',
                  metrics=['accuracy'])

    x_train = [[-1, -1],
               [-1, +1],
               [+1, -1],
               [+1, +1]]
    y_train = [0, 1, 1, 0]
    #model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=False)
    #model.save_weights(os.path.join(SCRIPT_DIR, 'bnn_qdense_bn_sign_act.h5'))
    model.load_weights(os.path.join(SCRIPT_DIR, 'bnn_qdense_bn_sign_act.h5'))
    return model


@pytest.fixture(scope='session')
def bnn_simple_model() -> tf.keras.Model:
    w1 = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    b1 = np.array([1, 2, 0, 1])
    w2 = np.array([-1, 1, -1, -1]).reshape(4, 1)
    b2 = np.array([1])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(qkeras.binary(alpha=1))(x)
    x = qkeras.QDense(4, kernel_quantizer=qkeras.binary(alpha=1), activation='binary')(x)
    x = qkeras.QDense(1, kernel_quantizer=qkeras.binary(alpha=1), activation='binary')(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[3].set_weights([w2, b2])
    return model


@pytest.fixture(scope='session')
def bnn_simple_bweight_model() -> tf.keras.Model:
    w1 = np.array([[1, -1], [-1, -1], [1, 1]])
    b1 = np.array([2, 0])

    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(qkeras.quantized_bits(bits=8, integer=8, keep_negative=False))(x)
    x = qkeras.QDense(2, kernel_quantizer=qkeras.binary(alpha=1), activation='binary')(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    return model


@pytest.fixture(scope='session')
def bnn_mnist_model() -> tf.keras.Model:
    """
        Build a dense binarized model in qkeras that uses a single integer layer at the start, and batch-norm
        layers in between the dense layers. The test only checks that verilog file was succesfully generated.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28*28
    num_classes = 10  # ten unique digits
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    y_train = tf.one_hot(y_train, 10)
    y_train = np.where(y_train < 0.1, -1., 1.)
    y_test = tf.one_hot(y_test, 10)
    y_test = np.where(y_test < 0.1, -1., 1.)

    model = tf.keras.models.Sequential()
    # We don't loose any info here since mnist are 8-bit gray-scale images. we just add this quantization
    # to explicitly encode this for the chisel4ml optimizer.
    model.add(tf.keras.layers.Input(shape=image_vector_size))
    model.add(qkeras.QActivation(qkeras.quantized_bits(bits=8, integer=8, keep_negative=False)))
    model.add(qkeras.QDense(64, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(qkeras.QDense(64, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(qkeras.QDense(64, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(qkeras.QDense(num_classes, kernel_quantizer=qkeras.binary(alpha=1), activation='binary'))

    model.compile(optimizer="adam",
                  loss='squared_hinge',
                  metrics=['accuracy'])

    #model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=False)
    #model.save_weights(os.path.join(SCRIPT_DIR, 'bnn_mnist_model.h5'))
    model.load_weights(os.path.join(SCRIPT_DIR, 'bnn_mnist_model.h5'))
    return model
