# models.__init__.py
__all__ = ["factory_method"]

import os
import importlib

from .keras_lbir_transform import KerasLbirTransform

__KERAS_TRANSFORM_DICT__ = dict()

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

