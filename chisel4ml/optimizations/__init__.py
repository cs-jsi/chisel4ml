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
# models.__init__.py
__all__ = ["qkeras_opt_list, register_qkeras_optimization"]

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
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("chisel4ml.optimizations." + module_name)

    qkeras_opt_list = sorted(qkeras_opt_list, key=lambda x: x.order)  # type: ignore
