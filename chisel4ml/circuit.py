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

import numpy as np

import chisel4ml.lbir.lbir_pb2 as lbir
import chisel4ml.lbir.services_pb2 as services
from chisel4ml import transforms
from chisel4ml.chisel4ml_server import start_server_once

log = logging.getLogger(__name__)


class Circuit:
    """Container class for the generated circuits. Objects of type Circuit hold a
    reference to the circuit implemnentation in the chisel4ml server memory. It also
    provides a python interface to that server via gRPC (via __call__ or predict).
    """

    def __init__(self, circuitId: int, input_quantizer, input_qtensor: lbir.QTensor):
        assert circuitId >= 0, (
            "Invalid circuitId provided. This parameter should be positive, but is"
            f" {circuitId}."
        )
        self.circuitId = circuitId
        self.input_quantizer = input_quantizer
        self.input_qtensor = input_qtensor
        self._server = start_server_once()

    def __call__(self, np_arr, sim_timeout_sec=100):
        "Simulate the circuit, timeout in seconds."
        qtensor = transforms.numpy_transforms.numpy_to_qtensor(
            np_arr, self.input_quantizer, self.input_qtensor
        )
        run_sim_params = services.RunSimulationParams(
            circuitId=self.circuitId, inputs=[qtensor]
        )
        run_sim_return = self._server.send_grpc_msg(run_sim_params)
        return np.array(run_sim_return.values[0].values)

    def predict(self, np_arr):
        self.__call__(np_arr)
