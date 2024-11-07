import os

import brevitas.nn as qnn
import numpy as np
import pytest
import torch
from pytest_cases import get_current_cases
from pytest_cases import parametrize_with_cases
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from torch.nn import Module

from chisel4ml import generate
from chisel4ml import optimize
from chisel4ml.utils import get_submodel
from tests.brevitas_quantizers import CommonWeightQuant
from tests.brevitas_quantizers import IntActQuant
from tests.brevitas_quantizers import IntBiasQuant
from tests.conftest import TEST_MODELS_LIST

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")


@parametrize_with_cases("model_data_info", cases=TEST_MODELS_LIST, has_tag="trainable")
def test_trainable_simulation(request, c4ml_server, model_data_info):
    model, data, training_info = model_data_info
    filename = get_current_cases(request)["model_data_info"][0]
    if request.config.getoption("--retrain"):
        print(f"Retraining model {filename}.")
        model.fit(
            data["X_train"],
            data["y_train"],
            batch_size=training_info["batch_size"],
            epochs=training_info["epochs"],
            callbacks=training_info["callbacks"],
            verbose=True,
        )
        opt_model = optimize.qkeras_model(model)
        opt_model.compile(
            optimizer=model.optimizer.from_config(model.optimizer.get_config()),
            loss=model.loss,
            metrics=["accuracy"],
        )
        opt_model.fit(
            data["X_train"],
            data["y_train"],
            batch_size=training_info["batch_size"],
            epochs=training_info["epochs"],
            callbacks=training_info["callbacks"],
            verbose=True,
        )
        if request.config.getoption("--save-retrained"):
            print(f"Saving model {filename} to {filename}.h5.")
            opt_model.save_weights(os.path.join(MODEL_DIR, f"{filename}.h5"))
    else:
        model_weights = os.path.join(MODEL_DIR, f"{filename}.h5")
        print(f"Loading weights from: {model_weights}.")
        opt_model = optimize.qkeras_model(model)
        opt_model.load_weights(model_weights)

    accelerators, lbir_model = generate.accelerators(
        opt_model, minimize="area", debug=request.config.getoption("--debug-trans")
    )
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server
    )
    assert circuit is not None
    for data in data["X_test"]:
        sw_res = opt_model.predict(np.expand_dims(data, axis=0))
        if isinstance(sw_res, tuple):
            sw_res = sw_res[0]
        hw_res = circuit(data)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


@parametrize_with_cases(
    "model_data_info", cases=TEST_MODELS_LIST, has_tag="trainable-gen"
)
def test_trainable_gen_simulation(request, c4ml_server, model_data_info):
    model, data, training_info = model_data_info
    filename = get_current_cases(request)["model_data_info"][0]
    if request.config.getoption("--retrain"):
        print(f"Retraining model {filename}.")
        model.fit(
            x=data["train_set"]
            .batch(training_info["batch_size"], drop_remainder=True)
            .repeat(training_info["epochs"]),
            validation_data=data["val_set"]
            .batch(training_info["batch_size"], drop_remainder=True)
            .repeat(training_info["epochs"]),
            batch_size=training_info["batch_size"],
            steps_per_epoch=int(
                training_info["train_len"] / training_info["batch_size"]
            ),
            validation_steps=int(
                training_info["val_len"] / training_info["batch_size"]
            ),
            epochs=training_info["epochs"],
            callbacks=training_info["callbacks"],
            verbose=True,
        )
        opt_model = optimize.qkeras_model(model)
        opt_model.compile(
            optimizer=model.optimizer.from_config(model.optimizer.get_config()),
            loss=model.loss,
            metrics=["accuracy"],
        )
        opt_model.fit(
            x=data["train_set"]
            .batch(training_info["batch_size"], drop_remainder=True)
            .repeat(training_info["epochs"]),
            validation_data=data["val_set"]
            .batch(training_info["batch_size"], drop_remainder=True)
            .repeat(training_info["epochs"]),
            batch_size=training_info["batch_size"],
            steps_per_epoch=int(
                training_info["train_len"] / training_info["batch_size"]
            ),
            validation_steps=int(
                training_info["val_len"] / training_info["batch_size"]
            ),
            epochs=training_info["epochs"],
            callbacks=training_info["callbacks"],
            verbose=True,
        )
        if request.config.getoption("--save-retrained"):
            print(f"Saving model {filename} to {filename}.h5.")
            opt_model.save_weights(os.path.join(MODEL_DIR, f"{filename}.h5"))
    else:
        model_weights = os.path.join(MODEL_DIR, f"{filename}.h5")
        print(f"Loading weights from: {model_weights}.")
        opt_model = optimize.qkeras_model(model)
        opt_model.load_weights(model_weights)

    accelerators, lbir_model = generate.accelerators(
        opt_model, minimize="area", debug=request.config.getoption("--debug-trans")
    )
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server
    )
    assert circuit is not None
    for x, _ in data["test_set"]:
        sw_res = opt_model.predict(np.expand_dims(x, axis=0))[0]
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


@parametrize_with_cases("model_data", cases=TEST_MODELS_LIST, has_tag="non-trainable")
def test_simulation(request, c4ml_server, model_data):
    (
        model,
        data,
    ) = model_data
    opt_model = optimize.qkeras_model(model)
    accelerators, lbir_model = generate.accelerators(
        opt_model, minimize="area", debug=request.config.getoption("--debug-trans")
    )
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server
    )
    if request.config.getoption("--num-layers") is not None:
        opt_model = get_submodel(opt_model, request.config.getoption("--num-layers"))
    assert circuit is not None
    for x in data:
        sw_res = opt_model.predict(np.expand_dims(x, axis=0))
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


@parametrize_with_cases("model_ishape_data", cases=TEST_MODELS_LIST, has_tag="brevitas")
def test_brevitas(request, c4ml_server, model_ishape_data):
    (
        model,
        ishape,
        data,
    ) = model_ishape_data
    accelerators, lbir_model = generate.accelerators(
        model,
        ishape=ishape,
        minimize="area",
        debug=request.config.getoption("--debug-trans"),
    )

    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server
    )
    assert circuit is not None
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


def get_conv_layer_model(
    input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq, depthwise=False
):
    class ConvLayerModel(Module):
        def __init__(self):
            super(ConvLayerModel, self).__init__()
            self.ishape = (1, input_ch, 4, 4)
            self.conv = qnn.QuantConv2d(
                in_channels=input_ch,
                out_channels=output_ch,
                groups=output_ch if depthwise else 1,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=wq,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if wq == 1 else 2 ** (wq - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=bq,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bq - 1) - 1,
                input_quant=IntActQuant,
                input_bit_width=iq,
                input_scaling_impl_type="const",
                input_scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
                output_quant=IntActQuant,
                output_bit_width=oq,
                output_scaling_impl_type="const",
                output_scaling_init=1 if oq == 1 else 2 ** (oq - 1) - 1,
            )

        def forward(self, x):
            return self.conv(x)

    model = ConvLayerModel()
    wshape = (output_ch, 1 if depthwise else input_ch, kernel_size[0], kernel_size[1])
    bshape = (output_ch,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape)
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.conv.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.conv.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data


@pytest.mark.parametrize("input_ch", (1, 3))
@pytest.mark.parametrize("output_ch", (1, 3))
@pytest.mark.parametrize("kernel_size", ((3, 3), (2, 3)))
@pytest.mark.parametrize("padding", (0,))
@pytest.mark.parametrize("iq", (1, 3))
@pytest.mark.parametrize("wq", (1, 5))
@pytest.mark.parametrize("bq", (3, 8))
@pytest.mark.parametrize("oq", (1, 4))
def test_combinational_conv(
    request, c4ml_server, input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq
):
    model, data = get_conv_layer_model(
        input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq
    )
    accelerators, lbir_model = generate.accelerators(
        model,
        ishape=model.ishape,
        minimize="delay",
        debug=request.config.getoption("--debug-trans"),
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
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


@pytest.mark.parametrize("input_ch", (3,))
@pytest.mark.parametrize("output_ch", (3,))
@pytest.mark.parametrize("kernel_size", ((3, 3), (2, 3)))
@pytest.mark.parametrize("padding", (0,))
@pytest.mark.parametrize("iq", (1, 3))
@pytest.mark.parametrize("wq", (1, 5))
@pytest.mark.parametrize("bq", (3, 8))
@pytest.mark.parametrize("oq", (1, 4))
def test_combinational_conv_dw(
    request, c4ml_server, input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq
):
    model, data = get_conv_layer_model(
        input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq, depthwise=True
    )
    accelerators, lbir_model = generate.accelerators(
        model,
        ishape=model.ishape,
        minimize="delay",
        debug=request.config.getoption("--debug-trans"),
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
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


def get_maxpool_layer_model(channels, input_size, kernel_size, padding, iq):
    class MaxPoolLayerModel(Module):
        def __init__(self, input_size):
            super(MaxPoolLayerModel, self).__init__()
            self.ishape = (1, channels) + input_size
            self.in_quant = qnn.QuantIdentity(
                act_quant=IntActQuant,
                bit_width=iq,
                scaling_impl_type="const",
                scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
            )
            self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, padding=padding)
            self.out_quant = qnn.QuantIdentity(
                act_quant=IntActQuant,
                bit_width=iq,
                scaling_impl_type="const",
                scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
            )

        def forward(self, x):
            tmp = self.in_quant(x)
            tmp = self.maxpool(tmp)
            return self.out_quant(tmp)

    model = MaxPoolLayerModel(input_size)
    # set seed for repeatability
    np.random.seed(42)
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data


@pytest.mark.parametrize("input_size", ((4, 4), (8, 15)))
@pytest.mark.parametrize(
    "channels",
    (
        1,
        3,
    ),
)
@pytest.mark.parametrize("kernel_size", ((3, 3), (2, 3)))
@pytest.mark.parametrize("padding", (0,))
@pytest.mark.parametrize("iq", (1, 3))
def test_combinational_maxpool(
    request, c4ml_server, input_size, channels, kernel_size, padding, iq
):
    model, data = get_maxpool_layer_model(
        channels, input_size, kernel_size, padding, iq
    )
    accelerators, lbir_model = generate.accelerators(
        model,
        ishape=model.ishape,
        minimize="delay",
        debug=request.config.getoption("--debug-trans"),
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
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()

def get_linear_layer_model(in_features, out_features, bias, iq, wq, bq, oq):
    class LinearLayerModel(Module):
        def __init__(self):
            super(LinearLayerModel, self).__init__()
            self.ishape = (1, in_features)
            self.linear = qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                weight_quant=CommonWeightQuant,
                weight_bit_width=wq,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if wq == 1 else 2 ** (wq - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=bq,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bq - 1) - 1,
                input_quant=IntActQuant,
                input_bit_width=iq,
                input_scaling_impl_type="const",
                input_scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
                output_quant=IntActQuant,
                output_bit_width=oq,
                output_scaling_impl_type="const",
                output_scaling_init=1 if oq == 1 else 2 ** (oq - 1) - 1,
            )

        def forward(self, x):
            return self.linear(x)

    model = LinearLayerModel()
    wshape = (out_features, in_features)
    bshape = (out_features,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape)
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.linear.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.linear.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data

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
    model, data = get_linear_layer_model(
        in_features, out_features, bias, iq, wq, bq, oq
    )
    accelerators, lbir_model = generate.accelerators(
        model,
        ishape=model.ishape,
        minimize="delay",
        debug=request.config.getoption("--debug-trans"),
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
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()
