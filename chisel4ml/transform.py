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
import tensorflow as tf

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.transforms import qkeras_trans_list
from chisel4ml.utils import clone_model_from_config


def qkeras_to_lbir(
    model: tf.keras.Model, name="chisel4ml_model", custom_trans_list=[], debug=False
) -> lbir.Model:
    "Applys transformation to a Keras model, and returns a LBIR model."
    model_copy = clone_model_from_config(model)
    lbir_model = lbir.Model()
    lbir_model.name = name
    xlayers = model_copy.layers
    trans_list = qkeras_trans_list if len(custom_trans_list) == 0 else custom_trans_list
    for trans in trans_list:
        if debug:
            print(_stringfy_layers(xlayers, trans))
        left = 0
        right = trans.num_layers
        while right <= len(xlayers):
            assert right > left
            if trans.is_applicable(xlayers[left:right]):
                xlayers[left:right] = trans(xlayers[left:right])
            else:
                left = left + 1
                right = right + 1
    for layer in xlayers:
        # Is there a better way to check if a layer is a member of oneof field?
        assert hasattr(layer, "__module__") and layer.__module__ == "lbir_pb2", (
            "Transformation to lbir model failed. Not all layers were able to be "
            f"transformed to lbir layers. Layers {layer} seems to be the problem."
            f"All the layers are {_stringfy_layers(xlayers, None)}"
        )
    lbir_model.layers.extend(xlayers)
    return lbir_model


def _stringfy_layers(layers, trans):
    temp = f"Printing layer status before applying trans: {type(trans)}.\n"
    for lay in layers:
        temp = f"{temp}{type(lay)}\n"
    return temp
