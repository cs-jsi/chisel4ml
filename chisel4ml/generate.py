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

from chisel4ml import chisel4ml_server
from chisel4ml import transform
from chisel4ml.circuit import Circuit
from chisel4ml.lbir.lbir_pb2 import Datatype
from chisel4ml.lbir.lbir_pb2 import Layer
from chisel4ml.lbir.lbir_pb2 import PreprocessLayer
from chisel4ml.lbir.lbir_pb2 import QTensor
from chisel4ml.lbir.services_pb2 import GenerateCircuitParams
from chisel4ml.lbir.services_pb2 import GenerateCircuitReturn
from chisel4ml.lbir.services_pb2 import LayerOptions
from chisel4ml.transforms.qkeras_util import get_input_quantization

log = logging.getLogger(__name__)


def circuit(
    opt_model: tf.keras.Model,
    get_mfcc=False,
    is_simple=False,
    pipeline=False,
    use_verilator=False,
    gen_vcd=False,
    gen_timeout_sec=600,
    axi_stream_width=None,
):
    assert gen_timeout_sec > 5, "Please provide at least a 5 second generation timeout."
    # TODO - add checking that the opt_model is correct
    # opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras_to_lbir(opt_model)
    if lbir_model is None:
        return None
    if get_mfcc:
        lbir_model.layers.insert(
            0,
            Layer(
                ltype=Layer.Type.PREPROC,
                input=QTensor(
                    dtype=Datatype(
                        quantization=Datatype.QuantizationType.UNIFORM,
                        signed=False,
                        bitwidth=13,
                        shift=[0],
                        offset=[0],
                    ),
                    shape=[512],  # KERNEL, CH, WIDTH, HEIGHT
                ),
                output=QTensor(
                    dtype=Datatype(
                        quantization=Datatype.QuantizationType.UNIFORM,
                        signed=True,
                        bitwidth=6,
                        shift=[0],
                        offset=[0],
                    ),
                    shape=[20],  # KERNEL, CH, WIDTH, HEIGHT
                ),
                preprocess_layer=PreprocessLayer(
                    ptype=PreprocessLayer.Type.MFCC,
                    fft_size=512,
                    step_size=512,
                    num_mels=20,
                    num_frames=32,
                ),
            ),
        )

    server = chisel4ml_server.start_server_once()
    gen_circt_ret = server.send_grpc_msg(
        GenerateCircuitParams(
            model=lbir_model,
            options=GenerateCircuitParams.Options(
                is_simple=is_simple,
                pipeline_circuit=pipeline,
                layers=generate_layer_options(lbir_model, axi_stream_width),
            ),
            use_verilator=use_verilator,
            gen_vcd=gen_vcd,
            generation_timeout_sec=gen_timeout_sec,
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

    circuit = Circuit(
        gen_circt_ret.circuit_id,
        # TODO: this is temporary
        tf.keras.activations.linear if get_mfcc else get_input_quantization(opt_model),
        lbir_model.layers[0].input,
    )
    return circuit


def generate_layer_options(lbir_model, axi_stream_width):
    options = []
    for layer in lbir_model.layers:
        if layer.ltype is Layer.Type.PREPROC:
            options.append(LayerOptions(bus_width_in=13, bus_width_out=6))
        else:
            options.append(LayerOptions(bus_width_in=32, bus_width_out=32))
    return options
