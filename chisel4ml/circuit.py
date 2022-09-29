import logging
import numpy as np

import chisel4ml.lbir.services_pb2 as services
import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.chisel4ml_server import start_server_once
from chisel4ml import transforms

log = logging.getLogger(__name__)


class Circuit:
    """
        Container class for the generated circuits. Objects of type Circuit hold a reference to the
        circuit implemnentation in the chisel4ml server memory. It also provides a python interface to
        that server via gRPC (via __call__ or predict).
    """
    def __init__(self, circuitId: int, input_quantizer, input_qtensor: lbir.QTensor):
        assert circuitId >= 0, f"Invalid circuitId provided. This parameter should be positive, but is {circuitId}."
        self.circuitId = circuitId
        self.input_quantizer = input_quantizer
        self.input_qtensor = input_qtensor
        self._server = start_server_once()

    def __call__(self, np_arr):
        qtensor = transforms.numpy_transforms.numpy_to_qtensor(np_arr,
                                                               self.input_quantizer,
                                                               self.input_qtensor)
        run_sim_params = services.RunSimulationParams(circuitId=self.circuitId, inputs=[qtensor])
        run_sim_return = self._server.send_grpc_msg(run_sim_params)
        return np.array(run_sim_return.values[0].values)

    def predict(self, np_arr):
        self.__call__(np_arr)
