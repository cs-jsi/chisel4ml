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

log = logging.getLogger(__name__)


class QKerasTransform(ABC):
    """
    QKeras transformation returns a sequence of lbir layers posibly, mixed with keras
    layers. The structure is similar to the QKerasOptimization class, but it seperated
    because optimizations must always return a valid QKeras object.
    """

    num_layers: int
    order: int = 0

    def __call__(self, layers):
        assert len(layers) == self.num_layers, (
            f"The number of layers for the {self.__class__} transformations should be"
            f" {self.num_layers}. The provided number of layers to the optimizer was"
            f" {len(layers)}."
        )
        log.info(
            f"Calling transformation {self.__class__} on"
            f" layers:{list(map(type, layers))}."
        )
        return self._call_impl(layers)

    @abstractmethod
    def _call_impl(self, layers):
        return []

    @abstractmethod
    def is_applicable(self, layers) -> bool:
        return False
