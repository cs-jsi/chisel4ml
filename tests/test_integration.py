from chisel4ml import generate

import os
import shutil
from pathlib import Path
import pytest


@pytest.mark.skip(reason='Not running while porting to sequential design.')
def test_qkeras_simple_dense_binarized_model_nofixedpoint(bnn_simple_model):
    """
        Build a fully dense binarized model in qkeras, and then runs it through chisel4ml to get an verilog processing
        pipeline. This test only checks that we are able to get an verilog file.
    """
    epp_handle = generate.circuit(bnn_simple_model)
    assert epp_handle is not None
    temp_path = str(Path('.', 'gen_temp').absolute())
    epp_handle.gen_hw(temp_path)
    assert any(f.endswith(".v") for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)


@pytest.mark.skip(reason='Not running while porting to sequential design.')
def test_qkeras_dense_binarized_fixedpoint_batchnorm(bnn_mnist_model):
    epp_handle = generate.circuit(bnn_mnist_model)
    assert epp_handle is not None
    temp_path = str(Path('.', 'gen_temp').absolute())
    epp_handle.gen_hw(temp_path)
    assert any(f.endswith(".v") for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)


@pytest.mark.skip(reason='Not running while porting to sequential design.')
def test_qkeras_sint_mnist_qdense_relu(sint_mnist_qdense_relu):
    epp_handle = generate.circuit(sint_mnist_qdense_relu)
    assert epp_handle is not None
    temp_path = str(Path('.', 'gen_temp').absolute())
    epp_handle.gen_hw(temp_path)
    assert any(f.endswith(".v") for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)
