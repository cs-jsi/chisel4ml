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

from chisel4ml.transforms import qkeras_trans_list
from chisel4ml.lbir.validate import is_valid_lbir_model
import chisel4ml.lbir.lbir_pb2 as lbir

import qkeras
import tensorflow as tf


def qkeras_to_lbir(model: tf.keras.Model, 
                   name="chisel4ml_model", 
                   custom_trans_list=[]) -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    model_copy = qkeras.utils.clone_model(model)
    lbir_model = lbir.Model()
    lbir_model.name = name
    xlayers = model_copy.layers
    trans_list = qkeras_trans_list if len(custom_trans_list) == 0 else custom_trans_list
    for trans in trans_list:
        l = 0
        r = trans.num_layers
        while r <= len(xlayers):
            assert r > l
            if trans.is_applicable(xlayers[l:r]):
                xlayers[l:r] = trans(xlayers[l:r])
            else:
                l = l + 1
                r = r + 1

    for layer in xlayers:
        assert isinstance(layer, lbir.Layer), (f"Transformation to lbir model failed. Not all layers were able to be " 
                                               f"transformed to lbir layers.")
    lbir_model.layers.extend(xlayers)
    assert is_valid_lbir_model(lbir_model)
    return lbir_model
