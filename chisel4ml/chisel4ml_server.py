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
import concurrent.futures
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

import chisel4ml.lbir.services_pb2 as services
import chisel4ml.lbir.services_pb2_grpc as services_grpc

log = logging.getLogger(__name__)

server = None
_custom_temp_dir = None


class Chisel4mlServer:
    """Handles the creation of a subprocess, it is used to safely start the chisel4ml
    server.
    """

    def __init__(self, command, temp_dir, host: str = "localhost", port: int = 50051):
        self._server_addr = host + ":" + str(port)
        self._channel = None
        self._stub = None
        self._temp_dir = temp_dir

        # We start a new instance of the server. It will check if there is an instance
        # already running, and if so will simply close itself.
        self._log_file = open(os.path.join(temp_dir, "chisel4ml_server.log"), "w+")
        self.task = subprocess.Popen(
            command + [temp_dir], stdout=self._log_file, stderr=self._log_file
        )
        log.info(f"Started task with pid: {self.task.pid}.")

        # We start a process to create the grpc stub (this can take some time).
        self._pool = ThreadPoolExecutor(1)
        self._future = self._pool.submit(self.create_grpc_channel)

        # Here we make sure that the chisel4ml server is shut down.
        atexit.register(self.stop)
        signal.signal(
            signal.SIGTERM, self.stop
        )  # This ensures kill pid also close the server.
        signal.signal(signal.SIGINT, self.stop)

    @property
    def temp_dir(self):
        return self._temp_dir

    @property
    def stdout(self):
        return self.task.stdout.read()

    @property
    def stderr(self):
        return self.task.stderr.read()

    def create_grpc_channel(self):
        self._channel = grpc.insecure_channel(self._server_addr)
        self._stub = services_grpc.Chisel4mlServiceStub(self._channel)
        log.info("Created grpc channel.")

    def send_grpc_msg(self, msg, timeout=480):
        concurrent.futures.wait([self._future])
        if isinstance(msg, services.GenerateCircuitParams):
            ret = self._stub.GenerateCircuit(msg, wait_for_ready=True, timeout=timeout)
        elif isinstance(msg, services.RunSimulationParams):
            ret = self._stub.RunSimulation(msg, wait_for_ready=True, timeout=timeout)
        elif isinstance(msg, services.DeleteCircuitParams):
            ret = self._stub.DeleteCircuit(msg, wait_for_ready=True, timeout=timeout)
        else:
            raise ValueError(f"Invalid msg to send via grpc. Message is of type {msg}.")

        return ret

    def is_running(self):
        if self.task is None:
            return False
        else:
            return self.task.poll() is None

    def stop(self):
        log.info(f"Stoping task with pid: {self.task.pid}.")
        if self._channel is not None:
            self._channel.close()
        self._log_file.close()
        self.task.terminate()


def start_server_once():
    global server
    if server is None:
        jar_file = Path(Path(__file__).parent, "bin", "chisel4ml.jar").resolve()
        if _custom_temp_dir is not None:
            temp_dir_root = _custom_temp_dir
        else:
            temp_dir_root = tempfile.gettempdir()
        temp_dir = Path(temp_dir_root, ".chisel4ml").resolve()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        # We move to the temp dir because chisels TargetDirAnotation works with
        # relative dirs, which can cause a problem on Windows, if your working dir is
        # not on the same disk as the temp_dir (can't get a proper relative directory)
        backup = os.getcwd()
        os.chdir(temp_dir)
        try:
            server = Chisel4mlServer(
                command=["java", "-jar", "-Xmx14g", str(jar_file)],
                temp_dir=str(temp_dir),
            )
        finally:
            os.chdir(backup)
    return server


def set_custom_temp_path(path):
    global _custom_temp_dir
    _custom_temp_dir = path
