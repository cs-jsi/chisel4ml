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

import os
import logging
from chisel4ml.circuit import Circuit
from chisel4ml.transforms.qkeras_util import get_input_quantization
from chisel4ml import chisel4ml_server, transform
from chisel4ml.lbir.services_pb2 import GenerateCircuitParams, GenerateCircuitReturn
from pathlib import Path
import tensorflow as tf

log = logging.getLogger(__name__)


def circuit(opt_model: tf.keras.Model, 
            is_simple=False, 
            use_verilator=False,
            gen_vcd=False,
            gen_timeout_sec=600):
    assert gen_timeout_sec > 5, "Please provide at least a 5 second generation timeout."
    # TODO - add checking that the opt_model is correct
    # opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras_to_lbir(opt_model)
    if lbir_model is None:
        return None

    server = chisel4ml_server.start_server_once()
    gen_circt_ret = server.send_grpc_msg(GenerateCircuitParams(model=lbir_model,
                                                               options=GenerateCircuitParams.Options(
                                                                            isSimple=is_simple),
                                                               useVerilator=use_verilator,
                                                               genVcd=gen_vcd,
                                                               generationTimeoutSec=gen_timeout_sec), 
                                         gen_timeout_sec + 2)
    if gen_circt_ret is None:
        return None
    elif gen_circt_ret.err.errId != GenerateCircuitReturn.ErrorMsg.SUCCESS:
        log.error(f"Circuit generation failed with error id:{gen_circt_ret.err.errId} and the following"
                  f" error message:{gen_circt_ret.err.msg}")
        return None

    circuit = Circuit(gen_circt_ret.circuitId,
                      get_input_quantization(opt_model),
                      lbir_model.layers[0].input)
    return circuit
