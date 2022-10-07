import os
import logging
from chisel4ml.circuit import Circuit
from chisel4ml import chisel4ml_server, transform
from chisel4ml.lbir.services_pb2 import GenerateCircuitParams, GenerateCircuitReturn
from pathlib import Path
import tensorflow as tf

log = logging.getLogger(__name__)


def circuit(opt_model: tf.keras.Model, 
            directory="./chisel4ml_circuit/", 
            is_simple=False, 
            use_verilator=False,
            writeVcd=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    # TODO - add checking that the opt_model is correct
    # opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras_to_lbir(opt_model)
    if lbir_model is None:
        return None

    relDir = Path(directory).absolute().relative_to(Path('.').absolute()).__str__()
    server = chisel4ml_server.start_server_once()
    gen_circt_ret = server.send_grpc_msg(GenerateCircuitParams(model=lbir_model,
                                                               options=GenerateCircuitParams.Options(
                                                                            isSimple=is_simple),
                                                               directory=relDir,
                                                               useVerilator=use_verilator))
    if gen_circt_ret is None:
        return None
    elif gen_circt_ret.err.errId != GenerateCircuitReturn.ErrorMsg.SUCCESS:
        log.error(f"Circuit generation failed with error id:{gen_circt_ret.err.errId} and the following"
                  f" error message:{gen_circt_ret.err.msg}")
        return None

    circuit = Circuit(gen_circt_ret.circuitId,
                      opt_model.layers[0].input_quantizer_internal,
                      lbir_model.layers[0].input)
    return circuit