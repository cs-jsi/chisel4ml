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

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


class FFTLayer(tf.keras.layers.Layer):
    """TODO"""

    def __init__(self, win_fn="hamming"):
        super(FFTLayer, self).__init__()
        self.frame_length = 512
        self.num_frames = 32
        self.sr = self.num_frames * self.frame_length  # approx 16000
        self.win_fn = win_fn
        self.window_fn = (
            np.hamming(self.frame_length)
            if win_fn == "hamming"
            else np.ones(self.frame_length)
        )

    @tf.function()
    def call(self, inputs):
        tensor = tf.numpy_function(self.np_call, [inputs], tf.float32, stateful=False)
        return tensor

    def np_call(self, inputs):
        res = np.fft.fft(inputs * self.window_fn, norm="backward", axis=-1).real
        return np.expand_dims(res, axis=-1).astype(np.float32)

    def get_config(self):
        base_config = super().get_config()
        config = {"win_fn": self.win_fn}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(win_fn=config["win_fn"])
