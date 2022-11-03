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


def qkeras_to_lbir(model: tf.keras.Model, name="chisel4ml_model") -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    orig_cfg = copy.deepcopy(model.get_config())

    lbir_model = lbir.Model()
    lbir_model.name = name
    for trans in qkeras_trans_list:
        l = 0
        r = trans.num_layers
        while r < len(xlayers):
            assert r > l
            if trans.is_applicable(xlayers[l:r]):
                xlayers[l:r] = trans(xlayers[l:r])
            else:
                l = l + 1
                r = r + 1
    new_cfg = model.get_config()
    # This makes sure we don't change the original model. Cloning the model, is causing wierd behavior so this design
    # was chosen.
    return orig_cfg, new_cfg
    #assert orig_cfg == new_cfg, "It appears that a transformation changed the original model. This is a sign of a bug."
    #for lay in layer:
    #    assert isinstance(lay, lbir.Layer)
    #lbir_model.layers.extend(xlayers)
    #assert is_valid_lbir_model(lbir_model)
    #return lbir_model
