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


class ProcessingElementCombToSeq:
    layers = ["conv2d", "maxpool2d", "dense"]
    numBeatsIn = None
    numBeatsOut = None
    area = 9999
    delay = 1


class FFTWrapper:
    layers = ["fft"]
    numBeatsIn = 1
    numBeatsOut = 1
    area = 40
    delay = 40


class LMFEWrapper:
    layers = ["lmfe"]
    numBeatsIn = 1
    numBeatsOut = 1
    area = 40
    delay = 40


class MaxPool2D:
    layers = ["maxpool2d"]
    area = 20
    delay = 20


class ProcessingElementSequentialConv:
    layers = ["conv2d"]
    area = 20
    delay = 20


ACCELERATORS = [
    ProcessingElementCombToSeq(),
    FFTWrapper(),
    LMFEWrapper(),
    MaxPool2D(),
    ProcessingElementSequentialConv(),
]
