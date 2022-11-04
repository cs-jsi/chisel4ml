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

from chisel4ml.optimizations import qkeras_opt_list
from tensorflow_model_optimization.python.core.sparsity.keras import prune

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import qkeras
import copy

import collections
import itertools
import logging
log = logging.getLogger(__name__)


def qkeras_model(model, skip_list=[]):
    "Applys optimizations to the model."
    new_model = prune.strip_pruning(model) 
    new_model = qkeras.utils.clone_model(new_model)

    for opt in qkeras_opt_list:
        if opt.__class__.__name__ in skip_list:
            continue
        l = 0
        r = opt.num_layers
        while r < len(new_model.layers):
            assert r > l
            if opt.is_applicable(new_model.layers[l:r]):
                new_model = opt(new_model, new_model.layers[l:r])
            else:
                l = l + 1
                r = r + 1

    return new_model
