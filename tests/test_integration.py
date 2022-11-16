from chisel4ml import generate, optimize

import os
import shutil
from pathlib import Path
import pytest


@pytest.mark.skip(reason="waiting for directory functionality to be added")
def test_qkeras_simple_dense_binarized_model_nofixedpoint(bnn_simple_model):
    """Generates a circuit from qkeras model, and then runs it through chisel4ml to get
    an verilog processing pipeline. This test only checks that we are able to get an
    verilog file.
    """
    opt_model = optimize.qkeras_model(bnn_simple_model)
    temp_path = str(Path(".", "gen_temp").absolute())
    circuit = generate.circuit(opt_model, directory=temp_path, is_simple=True)
    assert circuit is not None
    assert any((f.endswith(".v") or f.endswith(".sv")) for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)


@pytest.mark.skip(reason="waiting for directory functionality to be added")
def test_qkeras_dense_binarized_fixedpoint_batchnorm(bnn_mnist_model):
    opt_model = optimize.qkeras_model(bnn_mnist_model)
    temp_path = str(Path(".", "gen_temp").absolute())
    circuit = generate.circuit(opt_model, directory=temp_path, is_simple=True)
    assert circuit is not None
    assert any((f.endswith(".v") or f.endswith(".sv")) for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)


@pytest.mark.skip(reason="waiting for directory functionality to be added")
def test_qkeras_sint_mnist_qdense_relu(sint_mnist_qdense_relu):
    opt_model = optimize.qkeras_model(sint_mnist_qdense_relu)
    temp_path = str(Path(".", "gen_temp").absolute())
    circuit = generate.circuit(opt_model, directory=temp_path, is_simple=True)
    assert circuit is not None
    assert any((f.endswith(".v") or f.endswith(".sv")) for f in os.listdir(temp_path))
    shutil.rmtree(temp_path)
