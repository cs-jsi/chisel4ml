from chisel4ml import optimizer, transformer, generator
import tensorflow as tf
import qkeras
import pytest

import os
import os.path,subprocess
from subprocess import STDOUT,PIPE


def test_qkeras_simple_dense_binarized_model_nofixedpoint():
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    x = x_in = tf.keras.layers.Input(shape=5)
    x = qkeras.QDense(10, kernel_quantizer=qkeras.binary(alpha=1), use_bias=False)(x)
    x = qkeras.QDense(1, kernel_quantizer=qkeras.binary(alpha=1), use_bias=False)(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.compile()
    pbfile = "test_qkeras_dense.pb"
    vfile = "test_qkeras_dense.v"
    generator.generate_verilog(model, pbfile = pbfile)
    os.remove(pbfile)
    assert os.path.exists(vfile)
