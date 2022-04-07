# models.__init__.py
__all__ = ["keras_transform_factory"]

import os
import importlib
from typing import Dict

from tensorflow.keras.layers import Layer as KerasLayer
from chisel4ml.transforms.keras_lbir_transform import KerasLbirTransform
from chisel4ml.lbir_python.lbir_pb2 import Layer as LbirLayer

__KERAS_TRANSFORM_DICT__: Dict[KerasLayer, LbirLayer] = dict()


def keras_transform_factory(name):
    return __KERAS_TRANSFORM_DICT__[name]


def register_keras_transform(name):
    def register_keras_transform_fn(cls):
        if name in __KERAS_TRANSFORM_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, KerasLbirTransform):
            raise ValueError("Class %s is not a subclass of %s" % (cls, KerasLbirTransform))

        __KERAS_TRANSFORM_DICT__[name] = cls
        return cls

    return register_keras_transform_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('transforms.' + module_name)
