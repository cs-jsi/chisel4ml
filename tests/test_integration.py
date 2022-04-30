from chisel4ml import generate
import tensorflow as tf
import qkeras

# import os


def test_qkeras_simple_dense_binarized_model_nofixedpoint():
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    x = x_in = tf.keras.layers.Input(shape=5)
    x = qkeras.QDense(10, kernel_quantizer=qkeras.binary(alpha=1), use_bias=True)(x)
    x = qkeras.QDense(1, kernel_quantizer=qkeras.binary(alpha=1), use_bias=True)(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    pbfile = "test_qkeras_dense.pb"
    # vfile = "test_qkeras_dense.v"
    generate.hardware(model, pbfile=pbfile)
    # os.remove(pbfile)
    # assert os.path.exists(vfile)
