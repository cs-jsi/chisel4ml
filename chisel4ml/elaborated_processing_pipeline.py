import logging
import numpy as np

import chisel4ml.lbir.services_pb2 as services
from chisel4ml.chisel4ml_server import start_server_once
from chisel4ml import transforms

log = logging.getLogger(__name__)


class ElaboratedProcessingPipeline:
    """
        Container class for the elaborated processing pipeline (EPP). It holds a reference to the actual EPP
        in the chisel4ml server memory. It also provides a python interface to that server via gRPC.
    """
    def __init__(self, elab_res, input_quantizer):
        self.pp = elab_res.ppHandle
        self.input_quantizer = input_quantizer
        self._server = start_server_once()

    def __call__(self, np_arr):
        qtensor = transforms.numpy_transforms.numpy_to_qtensor(np_arr,
                                                               self.input_quantizer,
                                                               self.pp.input)
        pp_run_params = services.PpRunParams(ppHandle=self.pp, inputs=[qtensor])
        pp_run_return = self._server.send_grpc_msg(pp_run_params)
        return np.array(pp_run_return.values[0].values)

    def predict(self, np_arr):
        self.__call(np_arr)

    def gen_hw(self, gen_dir=''):

        gen_params = services.GenerateParams(name=self.pp.name, directory=gen_dir)
        return self._server.send_grpc_msg(gen_params)
