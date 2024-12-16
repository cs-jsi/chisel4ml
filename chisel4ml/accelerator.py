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
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AcceleratorData:
    name: str
    layers: List[str]
    area: int
    delay: int


ProcessingElementCombToSeq = AcceleratorData(
    name="ProcessingElementCombToSeq",
    layers=["conv2d", "maxpool2d", "dense"],
    area=9999,
    delay=1,
)

FFTWrapper = AcceleratorData(name="FFTWrapper", layers=["fft"], area=40, delay=40)

LMFEWrapper = AcceleratorData(name="LMFEWrapper", layers=["lmfe"], area=40, delay=40)

MaxPool2D = AcceleratorData(name="MaxPool2D", layers=["maxpool2d"], area=20, delay=20)

ProcessingElementSequentialConv = AcceleratorData(
    name="ProcessingElementSequentialConv", layers=["conv2d"], area=20, delay=20
)

ACCELERATORS: List[AcceleratorData] = [
    ProcessingElementCombToSeq,
    FFTWrapper,
    LMFEWrapper,
    MaxPool2D,
    ProcessingElementSequentialConv,
]
