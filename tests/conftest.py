import os
import socket
import subprocess
from pathlib import Path

import pytest

from chisel4ml.chisel4ml_server import Chisel4mlServer


def pytest_addoption(parser):
    parser.addoption(
        "--use-verilator",
        action="store_true",
        default=False,
        help="Should we use verilator for simulation?",
    )
    parser.addoption(
        "--gen-waveform",
        action="store_true",
        default=False,
        help="Generate a waveform file (fst)?",
    )
    parser.addoption(
        "--waveform-type",
        default="fst",
        type=str,
        help="Generate a fst or vcd? Only applies if --gen-waveform is present.",
    )
    parser.addoption(
        "--generation-timeout",
        default=600,
        type=int,
        help="How many seconds should the generation timeout?",
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Turn on visualization in certain tests.",
    )
    parser.addoption(
        "--debug-trans",
        action="store_true",
        default=False,
        help="Print debug information when performing transformations.",
    )
    parser.addoption(
        "--chisel4ml-jar",
        default=f"{os.path.dirname(__file__)}/../out/chisel4ml/assembly.dest/out.jar",
        type=str,
        help="Location of the chisel4ml jar to use for testing.",
    )


@pytest.fixture
def free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def c4ml_server(worker_id, request, free_port, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("chisel4ml") / worker_id
    c4ml_jar = Path(request.config.getoption("--chisel4ml-jar")).resolve()
    assert c4ml_jar.exists(), f"Path {c4ml_jar} does not exist."
    command = ["java", "-jar", f"{c4ml_jar}", "-p", f"{free_port}", "-d", f"{tmp_dir}"]
    c4ml_subproc = None
    c4ml_server = None
    try:
        c4ml_subproc = subprocess.Popen(command)
        c4ml_server = Chisel4mlServer(tmp_dir, free_port)
        yield c4ml_server
    finally:
        if c4ml_subproc is not None:
            c4ml_subproc.terminate()
        if c4ml_server is not None:
            c4ml_server.stop()


def generate_test_combiations():
    return [("my_accel1", 1), ("my_accel2", 2)], ["my_accel1_id", "mynto_accel2_id"]

def pytest_generate_tests(metafunc):
    if "accelerator_qonnx" in metafunc.fixturenames:
        generated_tests, generated_test_names = generate_test_combiations()
        metafunc.parametrize(
            argnames="accelerator_qonnx", 
            argvalues=generated_tests,
            ids=generated_test_names
        )