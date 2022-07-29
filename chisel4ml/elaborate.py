from chisel4ml import optimize, transform, chisel4ml_server
from chisel4ml.elaborated_processing_pipeline import ElaboratedProcessingPipeline
import tensorflow as tf


def qkeras_model(model: tf.keras.Model):
    opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras_to_lbir(opt_model)
    server = chisel4ml_server.start_server_once()
    elab_res = server.send_grpc_msg(lbir_model)
    epp_handle = ElaboratedProcessingPipeline(elab_res=elab_res,
                                              input_quantizer=opt_model.layers[0].input_quantizer_internal)
    return epp_handle
