from chisel4ml import elaborate
import chisel4ml.lbir.services_pb2 as services
import numpy as np


def test_compile_service(bnn_simple_model):
    """ Test if the compile service is working correctly. """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS


def test_run_service(bnn_simple_model):
    """ Tests if the run service (simulation) is working correctly). """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
    for i in [-1.0, 1.0]:
        for j in [-1.0, 1.0]:
            for k in [-1.0, 1.0]:
                sw_res = bnn_simple_model.predict(np.array([[i, j, k]]))
                hw_res = epp_handle(np.array([i, j, k]))
                assert sw_res == hw_res, \
                    f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
                    f"{hw_res}. Something is wrong here. The stated results are for the inputs {i}, {j}, {k}. "
