# models.__init__.py
__all__ = ["qkeras_opt_list"]

from chisel4ml.optimizations.qkeras_optimization import QKerasOptimization

import os
import importlib
from typing import List

qkeras_opt_list: List[QKerasOptimization] = list()


def register_qkeras_optimization(cls):
    if not issubclass(cls, QKerasOptimization):
        raise ValueError(f"Class {cls} is not a subclass of {QKerasOptimization}!")
    qkeras_opt_list.append(cls())
    return cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('chisel4ml.optimizations.' + module_name)

    qkeras_opt_list = sorted(qkeras_opt_list, key=lambda x: x.priority)  # type: ignore
