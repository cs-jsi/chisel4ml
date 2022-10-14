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

from chisel4ml.transforms import qkeras_transform_factory
import chisel4ml.lbir.lbir_pb2 as lbir

import tensorflow as tf


def qkeras_to_lbir(model: tf.keras.Model, name="chisel4ml_model") -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    lbir_model = lbir.Model()
    lbir_model.name = name
    for i, layer in enumerate(model.layers):
        lbir_layer = qkeras_transform_factory(layer)(layer)
        lbir_model.layers.extend([lbir_layer])
    return lbir_model
