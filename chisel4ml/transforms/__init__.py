# models.__init__.py
__all__ = ["qkeras_transform_factory"]

import os
import importlib
from typing import Dict

from tensorflow.keras.layers import Layer as KerasLayer
from chisel4ml.lbir.lbir_pb2 import Layer as LbirLayer

__QKERAS_TRANSFORM_DICT__: Dict[KerasLayer, LbirLayer] = dict()


def qkeras_transform_factory(keras_layer):
    assert keras_layer.__class__ in __QKERAS_TRANSFORM_DICT__, \
            f"Layer {keras_layer.__class__} is not (yet) supported by chisel4ml. Sorry about that."
    return __QKERAS_TRANSFORM_DICT__[keras_layer.__class__]


def register_qkeras_transform(keras_layer):
    def register_qkeras_transform_fn(fn):
        if keras_layer.__class__ in __QKERAS_TRANSFORM_DICT__:
            raise ValueError(f"Transform for {keras_layer} already registered!")
        __QKERAS_TRANSFORM_DICT__[keras_layer] = fn
        return fn

    return register_qkeras_transform_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('chisel4ml.transforms.' + module_name)
