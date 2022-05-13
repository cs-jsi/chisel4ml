from chisel4ml import optimize, transform, server_manager
import tensorflow as tf

import os
import logging

import grpc
import chisel4ml.lbir.services_pb2_grpc as services

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def qkeras_model(model: tf.keras.Model):
    opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras2lbir(opt_model)
    server_manager.start_chisel4ml_server_once()
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = services.PpServiceStub(channel)
        pp_handle = stub.Elaborate(lbir_model, wait_for_ready=True)

    return pp_handle
