from chisel4ml import generate
import tensorflow as tf
import numpy as np
import qkeras
import pytest

import os
import shutil
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


@pytest.mark.skip(reason="cant run this, need to kill the java program.")
def test_qkeras_simple_dense_binarized_model_nofixedpoint():
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    w1 = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    b1 = np.array([1, 2, 0, 0])
    w2 = np.array([-1, 1, -1, -1]).reshape(4, 1)
    b2 = np.array([0])
    x = x_in = tf.keras.layers.Input(shape=3)
    x = qkeras.QActivation(qkeras.binary(alpha=1))(x)
    x = qkeras.QDense(4, kernel_quantizer=qkeras.binary(alpha=1), activation='binary')(x)
    x = qkeras.QDense(1, kernel_quantizer=qkeras.binary(alpha=1), activation='binary')(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    model.layers[2].set_weights([w1, b1])
    model.layers[3].set_weights([w2, b2])
    generate.hardware(model, gen_dir=os.path.join(os.getcwd(), "gen"))
    assert any(f.endswith(".v") for f in os.listdir("./gen/"))
    shutil.rmtree("./gen/")


@pytest.mark.skip(reason="cant run this, need to kill the java program.")
def test_qkeras_dense_binarized_fixedpoint_batchnorm(bnn_mnist_model):
    # loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    generate.hardware(bnn_mnist_model, gen_dir=os.path.join(os.getcwd(), "gen"))
    assert any(f.endswith(".v") for f in os.listdir("./gen/"))
    shutil.rmtree("./gen/")
