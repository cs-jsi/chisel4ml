import chisel4ml as c4ml
import pytest

import os
import shutil
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


@pytest.mark.skip(reason="cant run this, need to kill the java program.")
def test_qkeras_simple_dense_binarized_model_nofixedpoint(bnn_simple_model):
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    pp_handle = c4ml.compile.qkeras_model(bnn_simple_model, gen_dir=os.path.join(os.getcwd(), "gen"))
    pp_handle.generate_verilog('./gen/')
    assert any(f.endswith(".v") for f in os.listdir("./gen/"))
    shutil.rmtree("./gen/")


@pytest.mark.skip(reason="cant run this, need to kill the java program.")
def test_qkeras_dense_binarized_fixedpoint_batchnorm(bnn_mnist_model):
    # loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    c4ml.generate.hardware(bnn_mnist_model, gen_dir=os.path.join(os.getcwd(), "gen"))
    assert any(f.endswith(".v") for f in os.listdir("./gen/"))
    shutil.rmtree("./gen/")
