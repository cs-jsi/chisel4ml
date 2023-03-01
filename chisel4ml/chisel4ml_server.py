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
import os
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threading import Event
from threading import Thread

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
        self._stub = None
        self.error = False
        # We start a new instance of the server. It will check if there is an instance
        # already running, and if so will simply close itself.
        self._log_file = open(os.path.join(temp_dir, "chisel4ml_server.log"), "w+")
        self._channel_initialized = Event()
        self.scala_thread = Thread(
            target=self.scala_thread_imp,
            args=(command, host + ":" + str(port), temp_dir),
        )
        self.scala_thread.start()

        # Here we make sure that the chisel4ml server is shut down.
        atexit.register(self.stop)
        signal.signal(
            signal.SIGTERM, self.stop
        )  # This ensures kill pid also close the server.
        signal.signal(signal.SIGINT, self.stop)

    def scala_thread_imp(self, command, server_addr, temp_dir):
        channel = grpc.insecure_channel(server_addr)
        self._stub = services_grpc.Chisel4mlServiceStub(channel)
        log.info("Created grpc channel.")
        self._channel_initialized.set()
        ret = subprocess.run(
            command + [temp_dir], stdout=self._log_file, stderr=self._log_file
        )
        if ret != 0:
            self.error = True
            raise RuntimeError("Chisel4ml server returned and error.")

    def send_grpc_msg(self, msg, timeout=60):
        self._channel_initialized.wait(timeout=10)
        if self.error:
            raise RuntimeError(
                f"Something is wrong with the chisel4ml server. Check "
                f"{self._log_file} for details."
            )
        if isinstance(msg, services.GenerateCircuitParams):
            fn = (
                self._stub.GenerateCircuit
            )  # (msg, wait_for_ready=True, timeout=timeout)
        elif isinstance(msg, services.RunSimulationParams):
            fn = self._stub.RunSimulation  # (msg, wait_for_ready=True, timeout=timeout)
        else:
            raise ValueError(f"Invalid msg to send via grpc. Message is of type {msg}.")
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, msg, wait_for_ready=True, timeout=timeout)
            while not future.done():
                time.sleep(1)
                if self.error:
                    raise RuntimeError(
                        f"The chisel4ml server crashed. Check {self._log_file} for "
                        f"details."
                    )
        return future.result()

    def stop(self):
        self._log_file.close()
        self.scala_thread.join()


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
        os.chdir(temp_dir)
        server = Chisel4mlServer(
            command=["java", "-jar", str(jar_file)], temp_dir=str(temp_dir)
        )
    return server


def set_custom_temp_path(path):
    global _custom_temp_dir
    _custom_temp_dir = path
