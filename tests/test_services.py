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
def test_trainable_simulation(request, model_data_info):
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

    accelerators, lbir_model = generate.accelerators(opt_model, minimize="area")
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
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
def test_trainable_gen_simulation(request, model_data_info):
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

    accelerators, lbir_model = generate.accelerators(opt_model, minimize="area")
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
    )
    assert circuit is not None
    for x, _ in data["test_set"]:
        sw_res = opt_model.predict(np.expand_dims(x, axis=0))[0]
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


@parametrize_with_cases("model_data", cases=TEST_MODELS_LIST, has_tag="non-trainable")
def test_simulation(request, model_data):
    (
        model,
        data,
    ) = model_data
    opt_model = optimize.qkeras_model(model)
    accelerators, lbir_model = generate.accelerators(opt_model, minimize="area")
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
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
def test_brevitas(request, model_ishape_data):
    (
        model,
        ishape,
        data,
    ) = model_ishape_data
    accelerators, lbir_model = generate.accelerators(
        model, ishape=ishape, minimize="area"
    )

    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
    )
    assert circuit is not None
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()


def get_conv_layer_model(input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq):
    class ConvLayerModel(Module):
        def __init__(self):
            super(ConvLayerModel, self).__init__()
            self.ishape = (1, input_ch, 4, 4)
            self.conv = qnn.QuantConv2d(
                in_channels=input_ch,
                out_channels=output_ch,
                groups=1,
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
    wshape = (output_ch, input_ch, kernel_size[0], kernel_size[1])
    bshape = (output_ch,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape)
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.conv.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.conv.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    # qonnx_model = transform.brevitas_to_qonnx(model, model.ishape)
    # qstr = f"i{iq}_w{wq}_b{bq}_o{oq}.onnx"
    # fname = f"conv_ich{input_ch}_och{output_ch}_ks{kernel_size}_p{padding}_{qstr}"
    # onnx.save(qonnx_model.model, fname)
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
    request, input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq
):
    model, data = get_conv_layer_model(
        input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq
    )
    accelerators, lbir_model = generate.accelerators(
        model, ishape=model.ishape, minimize="delay"
    )
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
    )
    assert circuit is not None
    for x in data:
        sw_res = (
            model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.delete_from_server()
