import atexit
import signal
import subprocess
import logging

from pathlib import Path

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import grpc
import chisel4ml.lbir.services_pb2_grpc as services_grpc
import chisel4ml.lbir.services_pb2 as services
import chisel4ml.lbir.lbir_pb2 as lbir

log = logging.getLogger(__name__)

server = None


class Chisel4mlServer:
    """ Handles the creation of a subprocess, it is used to safely start the chisel4ml server. """

    def __init__(self, command, host: str = 'localhost', port: int = 50051, grpc_timeout: int = 240):
        self._server_addr = host + ':' + str(port)
        self._channel = None
        self._stub = None
        self.GRPC_TIMEOUT = grpc_timeout
        self._log_file = open('chisel4ml_server.log', 'w')
        self.task = subprocess.Popen(command,
                                     stdout=self._log_file,
                                     stderr=self._log_file)
        log.info(f"Started task with pid: {self.task.pid}.")

        # We start a process to create the grpc stub (this can take some time).
        self._pool = ThreadPoolExecutor(1)
        self._future = self._pool.submit(self.create_grpc_channel)

        # Here we make sure that the chisel4ml server is shut down.
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, self.stop)  # This ensures kill pid also close the server.
        signal.signal(signal.SIGINT, self.stop)

    @property
    def stdout(self):
        return self.task.stdout.read()

    @property
    def stderr(self):
        return self.task.stderr.read()

    def create_grpc_channel(self):
        self._channel = grpc.insecure_channel(self._server_addr)
        self._stub = services_grpc.PpServiceStub(self._channel)
        log.info("Created grpc channel.")

    def send_grpc_msg(self, msg):
        concurrent.futures.wait([self._future])
        if isinstance(msg, lbir.Model):
            ret = self._stub.Elaborate(msg, wait_for_ready=True, timeout=self.GRPC_TIMEOUT)
        elif isinstance(msg, services.PpRunParams):
            ret = self._stub.Run(msg, wait_for_ready=True, timeout=self.GRPC_TIMEOUT)
        elif isinstance(msg, services.GenerateParams):
            ret = self._stub.Generate(msg, wait_for_ready=True, timeout=self.GRPC_TIMEOUT)
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
        self._channel.close()
        self._log_file.close()
        self.task.terminate()


def start_server_once():
    global server

    if server is None:
        server = Chisel4mlServer(command=['java', '-Xms6500M', '-jar', str(Path('bin', 'chisel4ml.jar'))])

    return server
