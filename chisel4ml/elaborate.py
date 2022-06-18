from chisel4ml import optimize, transform, server_manager, transforms
import tensorflow as tf
import numpy as np

import os
import logging

import grpc
import chisel4ml.lbir.services_pb2_grpc as services_grpc
import chisel4ml.lbir.services_pb2 as services

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

GRPC_TIMEOUT = 180  # a 3 minute timeout


class ElaboratedProcessingPipelineHandle:
    def __init__(self, pp, reply, input_quantizer):
        self.pp = pp
        self.reply = reply
        self.input_quantizer = input_quantizer

    def __call__(self, np_arr):
        qtensor = transforms.numpy_transforms.numpy_to_qtensor(np_arr,
                                                               self.input_quantizer,
                                                               self.pp.input)
        ppRunParams = services.PpRunParams(ppHandle=self.pp, inputs=[qtensor])
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = services_grpc.PpServiceStub(channel)
            pp_run_return = stub.Run(ppRunParams, wait_for_ready=True, timeout=GRPC_TIMEOUT)
        return np.array(pp_run_return.values[0].values)


def qkeras_model(model: tf.keras.Model):
    opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras2lbir(opt_model)
    server_manager.start_chisel4ml_server_once()
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = services_grpc.PpServiceStub(channel)
        elab_res = stub.Elaborate(lbir_model, wait_for_ready=True, timeout=GRPC_TIMEOUT)

    epp_handle = ElaboratedProcessingPipelineHandle(pp=elab_res.ppHandle,
                                                    reply=elab_res.reply,
                                                    input_quantizer=opt_model.layers[0].input_quantizer)
    return epp_handle
