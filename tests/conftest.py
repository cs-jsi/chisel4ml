import importlib
import os

from chisel4ml import chisel4ml_server


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
C4ML_SERVER = None
TEST_MODELS_DICT = {}
for _, _, files in os.walk(MODEL_DIR):
    python_files = [f for f in files if f.endswith(".py")]
    for file in python_files:
        filename, extension = file.split(".")
        if (
            filename not in TEST_MODELS_DICT
            and filename != "__init__"
            and extension == "py"
        ):
            module = importlib.import_module(f"tests.models.{filename}")
            f = getattr(module, f"case_{filename}")
            TEST_MODELS_DICT[filename] = f
            print(f"Added model: {filename} to TEST_MODELS_DICT.")
TEST_MODELS_LIST = list(TEST_MODELS_DICT.values())


def pytest_addoption(parser):
    parser.addoption(
        "--retrain",
        action="store_true",
        default=False,
        help="Should trainable models be retrained?",
    )
    parser.addoption(
        "--save-retrained",
        action="store_true",
        default=False,
        help="Should the retrained models be saved also?",
    )
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
        "--generation-timeout",
        default=600,
        type=int,
        help="How many seconds should the generation timeout?",
    )
    parser.addoption(
        "--num-layers",
        default=None,
        type=int,
        help="How many layers of the model to take? (default all)",
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Turn on visualization in certain tests.",
    )


def pytest_sessionstart(session):
    global C4ML_SERVER
    C4ML_SERVER = chisel4ml_server.connect_to_server()


def pytest_sessionfinish(session, exitstatus):
    global C4ML_SERVER
    if C4ML_SERVER is not None:
        C4ML_SERVER.stop()
