import copy

import numpy as np
import chisel4ml.lbir.lbir_pb2 as lbir


def numpy_to_qtensor(arr: np.array, input_quantizer, input_qtensor: lbir.QTensor) -> lbir.QTensor:
    assert np.array_equal(arr, input_quantizer(arr))
    assert list(arr.shape) == input_qtensor.shape

    qtensor = copy.deepcopy(input_qtensor)
    qtensor.values[:] = arr.tolist()
    return qtensor
