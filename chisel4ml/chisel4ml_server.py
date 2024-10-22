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
import atexit
import logging
import signal

import grpc

import chisel4ml
import chisel4ml.lbir.services_pb2 as services
import chisel4ml.lbir.services_pb2_grpc as services_grpc

log = logging.getLogger(__name__)
default_server = None


class Chisel4mlServer:
    """Handles the creation of a subprocess, it is used to safely start the chisel4ml
    server.
    """

    def __init__(self, temp_dir, port):
        self._server_addr = "localhost:" + str(port)
        self._channel = None
        self._stub = None
        self._temp_dir = temp_dir
        self._channel = grpc.insecure_channel(self._server_addr)
        self._stub = services_grpc.Chisel4mlServiceStub(self._channel)
        scala_version = self._stub.GetVersion(services.GetVersionParams()).version

        # this ignores the changes in a working repo (see _version.py)
        if (
            len(chisel4ml.__version_tuple__) == 5
            and "." in chisel4ml.__version_tuple__[4]
        ):
            log.warning("Using dirty repository!")
            vt = chisel4ml.__version_tuple__
            githash = vt[4].split(".")[0]
            python_version = f"{vt[0]}.{vt[1]}.{vt[2]}.{vt[3]}+{githash}"
        else:
            python_version = chisel4ml.__version__

        assert (
            python_version == scala_version
        ), f"Python/scala version missmatch: {python_version}/{scala_version}."
        log.info(f"Created grpc channel on {self._server_addr}.")

        # Here we make sure that the chisel4ml server is shut down.
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    @property
    def temp_dir(self):
        return self._temp_dir

    def stop(self):
        if self._channel is not None:
            self._channel.close()

    def send_grpc_msg(self, msg, timeout=480):
        if isinstance(msg, services.GenerateCircuitParams):
            ret = self._stub.GenerateCircuit(msg, wait_for_ready=True, timeout=timeout)
        elif isinstance(msg, services.RunSimulationParams):
            ret = self._stub.RunSimulation(msg, wait_for_ready=True, timeout=timeout)
        elif isinstance(msg, services.DeleteCircuitParams):
            ret = self._stub.DeleteCircuit(msg, wait_for_ready=True, timeout=timeout)
        else:
            raise ValueError(f"Invalid msg to send via grpc. Message is of type {msg}.")

        return ret


def connect_to_server(temp_dir="/tmp/.chisel4ml/", port: int = 50051):
    global default_server
    server = Chisel4mlServer(temp_dir=str(temp_dir), port=port)
    default_server = server
    return server
