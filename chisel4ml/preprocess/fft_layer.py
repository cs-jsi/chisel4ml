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

from chisel4ml.lbir.lbir_pb2 import FFTConfig

log = logging.getLogger(__name__)


class FFTLayer(tf.keras.layers.Layer):
    """TODO"""

    def __init__(self, cfg: FFTConfig):
        super(FFTLayer, self).__init__()
        self.cfg = cfg
        self.sr = cfg.num_frames * cfg.fft_size  # approx 16000
        self.window_fn = np.array(cfg.win_fn)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def call(self, inputs):
        tensor = tf.numpy_function(self.np_call, [inputs], tf.float32, stateful=False)
        return tf.reshape(
            tensor, (len(inputs), self.cfg.num_frames, self.cfg.fft_size, 1)
        )

    def np_call(self, inputs):
        results = []
        for x in inputs:
            res = np.fft.fft(x * self.window_fn, norm="backward", axis=-1).real
            results.append(np.expand_dims(res, axis=-1).astype(np.float32))
        return tf.convert_to_tensor(
            np.reshape(
                np.array(results),
                [len(inputs), self.cfg.num_frames, self.cfg.fft_size, 1],
            )
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"cfg": self.cfg}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(cfg=config["cfg"])
