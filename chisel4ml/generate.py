# Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import tensorflow as tf
import torch

from chisel4ml import chisel4ml_server
from chisel4ml import transform
from chisel4ml.circuit import Circuit
from chisel4ml.lbir.services_pb2 import GenerateCircuitParams
from chisel4ml.lbir.services_pb2 import GenerateCircuitReturn


log = logging.getLogger(__name__)


def circuit(
    model,
    ishape=None,
    is_simple=False,
    pipeline=False,
    use_verilator=False,
    gen_waveform=False,
    gen_timeout_sec=800,
    waveform_type="fst",
    num_layers=None,
    server=None,
    debug=False,
):
    assert gen_timeout_sec > 5, "Please provide at least a 5 second generation timeout."
    if isinstance(model, tf.keras.Model):
        qonnx_model = transform.qkeras_to_qonnx(model)
    elif isinstance(model, torch.nn.Module):
        qonnx_model = transform.brevitas_to_qonnx(model, ishape)
    else:
        raise TypeError(f"Model of type {type(model)} not supported.")
    lbir_model = transform.qonnx_to_lbir(qonnx_model, debug=debug)
    if lbir_model is None:
        return None
    if num_layers is not None:
        assert num_layers <= len(lbir_model.layers)
        for _ in range(len(lbir_model.layers) - num_layers):
            lbir_model.layers.pop()

    if server is None:
        if chisel4ml_server.default_server is None:
            server = chisel4ml_server.connect_to_server()
        else:
            server = chisel4ml_server.default_server

    gen_circt_ret = server.send_grpc_msg(
        GenerateCircuitParams(
            model=lbir_model,
            options=GenerateCircuitParams.Options(
                is_simple=is_simple,
                pipeline_circuit=pipeline,
            ),
            use_verilator=use_verilator,
            gen_waveform=gen_waveform,
            generation_timeout_sec=gen_timeout_sec,
            waveform_type=waveform_type,
        ),
        gen_timeout_sec + 2,
    )
    if gen_circt_ret is None:
        return None
    elif gen_circt_ret.err.err_id != GenerateCircuitReturn.ErrorMsg.SUCCESS:
        log.error(
            f"Circuit generation failed with error id:{gen_circt_ret.err.err_id} and"
            f" the following error message:{gen_circt_ret.err.msg}"
        )
        return None

    input_layer_type = lbir_model.layers[0].WhichOneof("sealed_value_optional")
    assert input_layer_type is not None
    input_qt = getattr(lbir_model.layers[0], input_layer_type).input
    circuit = Circuit(
        gen_circt_ret.circuit_id,
        input_qt,
        lbir_model,
        server,
    )
    return circuit
