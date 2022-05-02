from chisel4ml import generate
import tensorflow as tf
import numpy as np
import qkeras
from tensorflow.keras.datasets import mnist

import os
import shutil


def test_qkeras_simple_dense_binarized_model_nofixedpoint():
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    w1 = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    b1 = np.array([1, -1, 0, 0])
    w2 = np.array([-1, 1, -1, -1]).reshape(4, 1)
    b2 = np.array([0])
    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QDense(4, kernel_quantizer=qkeras.binary(alpha=1))(x)
    x = qkeras.QDense(1, kernel_quantizer=qkeras.binary(alpha=1))(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[1].set_weights([w1, b1])
    model.layers[2].set_weights([w2, b2])
    generate.hardware(model, gen_dir=os.path.join(os.getcwd(), "gen"))
    assert any(f.endswith(".v") for f in os.listdir("./gen/"))
    shutil.rmtree("./gen/")


def test_qkeras_dense_binarized_fixedpoint_batchnorm():
    """
        Build a dense binarized model in qkeras that uses a single fixed-point layer at the start, and batch-norm
        layers in between the dense layers. The test only checks that verilog file was succesfully generated.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28*28
    num_classes = 10  # ten unique digits
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

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
    model.add(qkeras.QDense(num_classes, kernel_quantizer=qkeras.binary(alpha=1), activation='softmax'))

    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=False)
    # loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    generate.hardware(model, gen_dir=os.path.join(os.getcwd(), "gen"))
    assert any(f.endswith(".v") for f in os.listdir("./gen/"))
    shutil.rmtree("./gen/")
