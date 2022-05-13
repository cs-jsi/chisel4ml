import atexit
import subprocess
import logging

from pathlib import Path

log = logging.getLogger(__name__)
server = None


class ServerManager:
    """ Handles the creation of a subprocess, it is used to safely start the chisel4ml server. """
    def __init__(self, command):
        self.task = None
        self.command = command

    @property
    def stdout(self):
        return self.task.stdout.read()

    @property
    def stderr(self):
        return self.task.stderr.read()

    def launch(self):
        self.task = subprocess.Popen(self.command,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        log.info(f"Started task with pid: {self.task.pid}.")
        atexit.register(self.stop)  # Here we make sure that the chisel4ml server is shut down.

    def is_running(self):
        if self.task is None:
            return False
        else:
            return self.task.poll() is None

    def stop(self):
        log.info(f"Stoping task with pid: {self.task.pid}.")
        try:
            self.task.terminate()
        except PermissionError:
            log.error("Permission Error!")


def start_chisel4ml_server_once():
    global server

    if server is None:
        server = ServerManager(["java", "-jar", str(Path("bin/chisel4ml.jar").absolute())])
        server.launch()
