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
import os
import shutil
from pathlib import Path

import numpy as np

import chisel4ml.lbir.services_pb2 as services
from chisel4ml import transforms
from chisel4ml.chisel4ml_server import Chisel4mlServer
from chisel4ml.chisel4ml_server import connect_to_server
from chisel4ml.lbir.qtensor_pb2 import QTensor

log = logging.getLogger(__name__)


class Circuit:
    """Container class for the generated circuits. Objects of type Circuit hold a
    reference to the circuit implemnentation in the chisel4ml server memory. It also
    provides a python interface to that server via gRPC (via __call__ or predict).
    """

    def __init__(
        self,
        circuit_id: int,
        input_quantizer,
        input_qtensor: QTensor,
        lbir_model,
        server: Chisel4mlServer = None,
    ):
        assert circuit_id >= 0, (
            "Invalid circuitId provided. This parameter should be positive, but is"
            f" {circuit_id}."
        )
        self.circuit_id = circuit_id
        self.input_quantizer = input_quantizer
        self.input_qtensor = input_qtensor
        if server is None:
            self._server = connect_to_server()
        else:
            self._server = server
        self.lbir_model = lbir_model
        self.consumed_cycles = None

    def __call__(self, np_arr, sim_timeout_sec=200):
        "Simulate the circuit, timeout in seconds."
        qtensors = transforms.numpy_transforms.numpy_to_qtensor(
            np_arr, self.input_quantizer, self.input_qtensor
        )
        run_sim_params = services.RunSimulationParams(
            circuit_id=self.circuit_id, inputs=qtensors
        )
        run_sim_return = self._server.send_grpc_msg(
            run_sim_params, timeout=sim_timeout_sec
        )
        self.consumed_cycles = run_sim_return.consumed_cycles
        results = []
        for res in run_sim_return.values:
            results.append(np.array(res.values).reshape(res.shape))
        return np.concatenate(results, axis=0)

    def predict(self, np_arr):
        return self.__call__(np_arr)

    def delete_from_server(self):
        delete_circuit_params = services.DeleteCircuitParams(circuit_id=self.circuit_id)
        delete_circuit_return = self._server.send_grpc_msg(
            delete_circuit_params, timeout=10
        )
        log.info(delete_circuit_return.msg)
        return delete_circuit_return.success

    def package(self, directory=None):
        if directory is None:
            raise ValueError("Directory parameter missing.")
        temp_dir = self._server.temp_dir
        temp_circuit_dir = os.path.join(temp_dir, f"circuit{self.circuit_id}")

        def get_files(extensions, directory):
            all_files = []
            for ext in extensions:
                all_files.extend(Path(directory).glob(ext))
            return all_files

        files = get_files(("*.sv", "*.bin", "*.hex"), directory=temp_circuit_dir)
        os.makedirs(Path(directory).absolute(), exist_ok=True)
        for file in files:
            dest_file = os.path.join(directory, os.path.split(file)[1])
            shutil.copyfile(file, dest_file)
