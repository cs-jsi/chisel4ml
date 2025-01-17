from brevitas.export import export_qonnx
import numpy as np
import pytest
import torch
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.cleanup import cleanup_model

from chisel4ml import generate
from chisel4ml import transform

from tests.brevitas_models import get_cnn_model
from tests.brevitas_models import get_conv_layer_model
from tests.brevitas_models import get_linear_layer_model
from tests.brevitas_models import get_maxpool_layer_model


@pytest.mark.parametrize("input_ch", (1, 3))
@pytest.mark.parametrize("output_ch", (1, 3))
@pytest.mark.parametrize("kernel_size", ((3, 3), (2, 3)))
@pytest.mark.parametrize("padding", (0, "same"))
@pytest.mark.parametrize("stride", (1, 2))
@pytest.mark.parametrize("iq", (1, 3))
@pytest.mark.parametrize("wq", (1, 5))
@pytest.mark.parametrize("bq", (3, 8))
@pytest.mark.parametrize("oq", (1, 4))
def test_combinational_conv(
    request,
    c4ml_server,
    input_ch,
    output_ch,
    kernel_size,
    padding,
    stride,
    iq,
    wq,
    bq,
    oq,
):
    if padding != 0 and iq == 1:
        # this is an illegal combination binary type has no zero to pad!
        return
    brevitas_model, data = get_conv_layer_model(
        input_ch, output_ch, kernel_size, padding, stride, iq, wq, bq, oq
    )
    qonnx_model = _brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    circuit = _qonnx_to_circuit(qonnx_model, request, c4ml_server)
    _compare_models(qonnx_model, circuit, data)


@pytest.mark.parametrize("input_ch", (3,))
@pytest.mark.parametrize("output_ch", (3,))
@pytest.mark.parametrize("kernel_size", ((3, 3), (2, 3)))
@pytest.mark.parametrize("padding", (0,))
@pytest.mark.parametrize("stride", (1,))
@pytest.mark.parametrize("iq", (1, 3))
@pytest.mark.parametrize("wq", (1, 5))
@pytest.mark.parametrize("bq", (3, 8))
@pytest.mark.parametrize("oq", (1, 4))
def test_combinational_conv_dw(
    request,
    c4ml_server,
    input_ch,
    output_ch,
    kernel_size,
    padding,
    stride,
    iq,
    wq,
    bq,
    oq,
):
    brevitas_model, data = get_conv_layer_model(
        input_ch,
        output_ch,
        kernel_size,
        padding,
        stride,
        iq,
        wq,
        bq,
        oq,
        depthwise=True,
    )
    qonnx_model = _brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    circuit = _qonnx_to_circuit(qonnx_model, request, c4ml_server)
    _compare_models(qonnx_model, circuit, data)


@pytest.mark.parametrize("input_size", ((4, 4), (8, 15)))
@pytest.mark.parametrize(
    "channels",
    (
        1,
        3,
    ),
)
@pytest.mark.parametrize("kernel_size", ((3, 3), (2, 3)))
@pytest.mark.parametrize("padding", (0, 1))
@pytest.mark.parametrize("stride", (1,))
@pytest.mark.parametrize("iq", (1, 3))
def test_combinational_maxpool(
    request, c4ml_server, input_size, channels, kernel_size, padding, stride, iq
):
    brevitas_model, data = get_maxpool_layer_model(
        channels, input_size, kernel_size, padding, stride, iq
    )
    qonnx_model = _brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    circuit = _qonnx_to_circuit(qonnx_model, request, c4ml_server)
    _compare_models(qonnx_model, circuit, data)


@pytest.mark.parametrize("in_features", (4, 8, 32))
@pytest.mark.parametrize("out_features", (4, 8, 32))
@pytest.mark.parametrize("bias", (True, False))
@pytest.mark.parametrize("iq", (1, 5))
@pytest.mark.parametrize("wq", (1, 5))
@pytest.mark.parametrize("bq", (3, 5))
@pytest.mark.parametrize("oq", (1, 3))
def test_combinational_fullyconnected(
    request, c4ml_server, in_features, out_features, bias, iq, wq, bq, oq
):
    brevitas_model, data = get_linear_layer_model(
        in_features, out_features, bias, iq, wq, bq, oq
    )
    qonnx_model = _brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    circuit = _qonnx_to_circuit(qonnx_model, request, c4ml_server)
    _compare_models(qonnx_model, circuit, data)


@pytest.mark.parametrize("in_features", (2,))
@pytest.mark.parametrize("out_features", (3,))
@pytest.mark.parametrize("bias", (True,))
@pytest.mark.parametrize("iq", (3,))
@pytest.mark.parametrize("wq", (3,))
@pytest.mark.parametrize("bq", (5,))
@pytest.mark.parametrize("oq", (3,))
@pytest.mark.parametrize("weight_scale", (0.25, 0.5, 2))
def test_combinational_fullyconnected_nonunitscale(
    request, c4ml_server, in_features, out_features, bias, iq, wq, bq, oq, weight_scale
):
    brevitas_model, data = get_linear_layer_model(
        in_features, out_features, bias, iq, wq, bq, oq, weight_scale
    )
    qonnx_model = _brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    circuit = _qonnx_to_circuit(qonnx_model, request, c4ml_server)
    _compare_models(qonnx_model, circuit, data)


@pytest.mark.parametrize("input_size", ((6, 6),))
@pytest.mark.parametrize("in_ch", (1, 3))
def test_combinational_cnn(request, c4ml_server, input_size, in_ch):
    brevitas_model, data = get_cnn_model(input_size, in_ch)
    qonnx_model = _brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    circuit = _qonnx_to_circuit(qonnx_model, request, c4ml_server)
    _compare_models(qonnx_model, circuit, data)


def _brevitas_to_qonnx(brevitas_model, input_shape):
    qonnx_proto =export_qonnx(brevitas_model, torch.randn(input_shape))
    qonnx_model = ModelWrapper(qonnx_proto)
    qonnx_model = cleanup_model(qonnx_model)
    return qonnx_model


def _qonnx_to_circuit(qonnx_model, request, c4ml_server):
    lbir_model = transform.qonnx_to_lbir(
        qonnx_model,
        debug=request.config.getoption("--debug-trans"),
    )
    accelerators = generate.accelerators(
        lbir_model,
        minimize="delay",
    )
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server,
    )
    assert circuit is not None
    return circuit


def _compare_models(qonnx_model, circuit, test_data):
    for x in test_data:
        expanded_x = np.expand_dims(x, axis=0)
        input_name = qonnx_model.model.graph.input[0].name
        qonnx_res = execute_onnx(qonnx_model, {input_name: expanded_x})
        qonnx_res = qonnx_res[list(qonnx_res.keys())[0]]
        hw_res = circuit(x)
        assert np.array_equal(hw_res.flatten(), qonnx_res.flatten())
    circuit.delete_from_server()


