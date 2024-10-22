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
import itertools
import logging

import numpy as np
import tensorflow as tf
import torch
from ortools.sat.python import cp_model

from chisel4ml import chisel4ml_server
from chisel4ml import transform
from chisel4ml.accelerator import ACCELERATORS
from chisel4ml.accelerator import ProcessingElementCombToSeq
from chisel4ml.circuit import Circuit
from chisel4ml.lbir.services_pb2 import Accelerator
from chisel4ml.lbir.services_pb2 import GenerateCircuitParams
from chisel4ml.lbir.services_pb2 import GenerateCircuitReturn

log = logging.getLogger(__name__)


class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, lbir_model, accelerators):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables = variables
        self.solution_list = []
        self.lbir_model = lbir_model
        self.accelerators = accelerators

    def on_solution_callback(self):
        solution_raw = np.zeros((len(self.lbir_model.layers), len(self.accelerators)))
        for lay, a in self.variables.keys():
            solution_raw[lay][a] = self.Value(self.variables[(lay, a)])
        solution = []
        for lay in solution_raw:
            acc_ind = lay.tolist().index(1)
            solution.append(self.accelerators[acc_ind])
        self.solution_list.append(solution)


def solution_to_accelerators(solution, lbir_layers):
    merged = []
    comb_list = []
    # merging ProcessingElementCombToSeq
    for ind, ls in enumerate(solution):
        if isinstance(ls, ProcessingElementCombToSeq):
            comb_list += (ls, lbir_layers[ind])
        else:
            if len(comb_list) > 0:
                merged += [comb_list] + [[(ls, lbir_layers[ind])]]
                comb_list = []
            else:
                merged += [[(ls, lbir_layers[ind])]]
    if len(comb_list) > 0:
        merged += [comb_list]
    acc_list = []
    for layer_list in merged:
        acc = Accelerator()
        acc.name = type(solution[0]).__name__
        for _, lbir_layer in layer_list:
            acc.layers.append(lbir_layer)
        acc_list.append(acc)
    return acc_list


def accelerators(model, ishape=None, num_layers=None, minimize="area", debug=False):
    if isinstance(model, tf.keras.Model):
        qonnx_model = transform.qkeras_to_qonnx(model)
    elif isinstance(model, torch.nn.Module):
        qonnx_model = transform.brevitas_to_qonnx(model, ishape)
    else:
        raise TypeError(f"Model of type {type(model)} not supported.")
    lbir_model = transform.qonnx_to_lbir(qonnx_model, debug=debug)
    if num_layers is not None:
        assert num_layers <= len(lbir_model.layers)
        for _ in range(len(lbir_model.layers) - num_layers):
            lbir_model.layers.pop()
    model_cp = cp_model.CpModel()
    vars_cp = {}

    for lay in range(len(lbir_model.layers)):
        for a in range(len(ACCELERATORS)):
            vars_cp[(lay, a)] = model_cp.new_bool_var(f"l{lay}-a{a}")

    # Only 1 accelerator per layer
    for lay in range(len(lbir_model.layers)):
        model_cp.add(sum(vars_cp[(lay, a)] for a in range(len(ACCELERATORS))) == 1)

    # Each accelerator only for layers it can handle
    for lay in range(len(lbir_model.layers)):
        for a in range(len(ACCELERATORS)):
            if (
                lbir_model.layers[lay].WhichOneof("sealed_value_optional")
                not in ACCELERATORS[a].layers
            ):
                model_cp.add(vars_cp[(lay, a)] == 0)

    la_iter = itertools.product(range(len(lbir_model.layers)), range(len(ACCELERATORS)))
    if minimize == "area":
        model_cp.minimize(
            sum(vars_cp[(lay, a)] * ACCELERATORS[a].area for lay, a in la_iter)
        )
    elif minimize == "delay":
        model_cp.minimize(
            sum(vars_cp[(lay, a)] * ACCELERATORS[a].delay for lay, a in la_iter)
        )
    else:
        raise NotImplementedError

    solver = cp_model.CpSolver()
    solution_collector = VarArraySolutionCollector(vars_cp, lbir_model, ACCELERATORS)
    solution_status = solver.solve(model_cp, solution_collector)
    assert solution_status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    solution = solution_collector.solution_list[0]
    accelerators = solution_to_accelerators(solution, lbir_model.layers)
    return accelerators, lbir_model


def circuit(
    accelerators,
    lbir_model,
    use_verilator=False,
    gen_waveform=False,
    gen_timeout_sec=800,
    waveform_type="fst",
    num_layers=None,
    server=None,
    debug=False,
):
    assert gen_timeout_sec > 5, "Please provide at least a 5 second generation timeout."
    if server is None:
        if chisel4ml_server.default_server is None:
            server = chisel4ml_server.connect_to_server()
        else:
            server = chisel4ml_server.default_server

    gen_circt_ret = server.send_grpc_msg(
        GenerateCircuitParams(
            accelerators=accelerators,
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

    input_layer_type = accelerators[0].layers[0].WhichOneof("sealed_value_optional")
    assert input_layer_type is not None
    input_qt = getattr(accelerators[0].layers[0], input_layer_type).input
    circuit = Circuit(
        gen_circt_ret.circuit_id,
        input_qt,
        lbir_model,
        server,
    )
    return circuit
