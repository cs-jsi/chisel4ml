# models.__init__.py
__all__ = ["qkeras_opt_factory"]

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization
from tensorflow.keras.layers import Layer as KerasLayer

import os
import importlib
from typing import Dict
from collections import defaultdict

class KeyDict(defaultdict):
    """ The defaultdict dictonary doesn't have the option to use arguments in the lambda function. When we encounter
        a missing key (unknown layer) we return a function, that returns that layer as is.
    """
    def __missing__(self, key):
        return lambda x: x
                
__QKERAS_OPT_DICT__: Dict[KerasLayer, QKerasOptimization] = KeyDict()


def qkeras_opt_factory(name):
    return __QKERAS_OPT_DICT__[name.__class__]


def register_qkeras_transform(class_list):
    if not isinstance(class_list, list):
        class_list = [class_list]

    def register_qkeras_transform_fn(cls):
        if not issubclass(cls, QKerasOptimization):
            raise ValueError(f"Class {cls} is not a subclass of {QKerasOptimization}!")
        for c in class_list:
            if c in __QKERAS_OPT_DICT__:
                raise ValueError(f"Class {c} already registered!")
            __QKERAS_OPT_DICT__[c] = cls()
        return cls

    return register_qkeras_transform_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('chisel4ml.optimizations.' + module_name)
