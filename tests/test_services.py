from chisel4ml import elaborate
import chisel4ml.lbir.services_pb2 as services

import os
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def test_compile_service(bnn_simple_model):
    """
        Test if the compile service is working correctly.
    """
    pp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert pp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
