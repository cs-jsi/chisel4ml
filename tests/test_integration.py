from chisel4ml import generate
import tensorflow as tf
import numpy as np
import qkeras

import os


def test_qkeras_simple_dense_binarized_model_nofixedpoint():
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    w1 = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    b1 = np.array([1, -1, 0, 0])
    w2 = np.array([-1, 1, -1, -1]).reshape(4,1)
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
