# Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import numpy as np

import chisel4ml.lbir.lbir_pb2 as lbir


def numpy_to_qtensor(
    arr: np.ndarray, input_quantizer, input_qtensor: lbir.QTensor
) -> list[lbir.QTensor]:
    assert np.array_equal(arr, input_quantizer(arr))
    if len(arr.shape) == 1:
        assert list(arr.shape) == input_qtensor.shape
        arr = arr.reshape([1, arr.shape[0]])
    else:
        assert len(arr.shape) == 2
        assert arr.shape[1] == input_qtensor.shape[0]

    results = []
    for tensor in arr:
        qtensor = copy.deepcopy(input_qtensor)
        qtensor.values[:] = tensor.tolist()
        results.append(qtensor)
    return results
