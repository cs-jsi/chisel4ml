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

from chisel4ml.lbir.qtensor_pb2 import QTensor


def numpy_to_qtensor(arr: np.ndarray, input_quantizer, input_qtensor: QTensor):
    """Converts numpy tensors to a list of qtensor objects."""
    if len(arr.shape) == len(input_qtensor.shape) + 1:
        narr = arr
    elif len(arr.shape) == len(input_qtensor.shape):
        narr = np.expand_dims(arr, axis=0)
    else:
        raise ValueError(
            f"Incompatible dimensions of the input array. Input array has shape "
            f"{arr.shape}, but input should be of shape {input_qtensor.shape} with "
            f"possible batch dimension."
        )
    assert np.array_equal(arr, input_quantizer(arr)), "Input is not properly quantized."

    results = []
    for tensor in narr:
        qtensor = copy.deepcopy(input_qtensor)
        qtensor.values[:] = tensor.flatten().tolist()
        results.append(qtensor)
    return results
