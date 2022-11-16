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
import logging
from abc import ABC
from abc import abstractmethod
from typing import Sequence

from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.keras.models import Model as KerasModel

log = logging.getLogger(__name__)


class QKerasOptimization(ABC):
    num_layers: int
    order: int = 0

    def __call__(self, model: KerasModel, layers: Sequence[KerasLayer]) -> KerasModel:
        assert len(layers) == self.num_layers, (
            f"The number of layers for the {self.__class__} optimizations should be"
            f" {self.num_layers}. The provided number of layers to the optimizer was"
            f" {len(layers)}."
        )
        log.info(f"Calling optimization {self.__class__} on layers:{layers}.")
        return self._call_impl(model, layers)

    @abstractmethod
    def _call_impl(self, model: KerasModel, layers: Sequence[KerasLayer]) -> KerasModel:
        return []

    @abstractmethod
    def is_applicable(self, layers: Sequence[KerasLayer]) -> bool:
        return False
