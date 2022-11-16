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

import chisel4ml.lbir.lbir_pb2 as lbir
import logging

log = logging.getLogger(__name__)

_exception_list = []


def is_valid_lbir_model(model):
    """Validates a lbir model to see if all of the fields that should be present are
    present."""
    is_valid = True
    is_valid = is_valid and isinstance(model, lbir.Model)
    is_valid = is_valid and model.name != ""
    is_valid = is_valid and len(model.layers) > 0
    for layer in model.layers:
        is_valid = is_valid and layer.HasField("thresh")
        is_valid = is_valid and layer.HasField("weights")
        is_valid = is_valid and layer.HasField("input")
        is_valid = is_valid and layer.HasField("output")

    return is_valid
