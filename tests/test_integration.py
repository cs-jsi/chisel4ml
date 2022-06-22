from chisel4ml import elaborate
import chisel4ml.lbir.services_pb2 as services

import pytest

import os
import shutil
from pathlib import Path


def test_qkeras_simple_dense_binarized_model_nofixedpoint(bnn_simple_model):
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
    temp_path = str(Path('.', 'gen_temp').absolute())
    epp_handle.gen_hw(temp_path)
    assert any(f.endswith(".v") for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)


@pytest.mark.skip(reason="cant run this, need to kill the java program.")
def test_qkeras_dense_binarized_fixedpoint_batchnorm(bnn_mnist_model):
    epp_handle = elaborate.qkeras_model(bnn_mnist_model)
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
    temp_path = str(Path('.', 'gen_temp').absolute())
    epp_handle.gen_hw(temp_path)
    assert any(f.endswith(".v") for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)
