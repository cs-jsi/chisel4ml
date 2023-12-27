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

import librosa
import numpy as np
import tensorflow as tf

from chisel4ml.lbir.lbir_pb2 import LMFEConfig

log = logging.getLogger(__name__)


class LMFELayer(tf.keras.layers.Layer):
    """TODO"""

    def __init__(self, cfg: LMFEConfig):
        super(LMFELayer, self).__init__()
        self.cfg = cfg
        self.sr = self.cfg.num_frames * self.cfg.fft_size  # approx 16000
        self.filter_banks = librosa.filters.mel(
            n_fft=self.cfg.fft_size,
            sr=self.sr,
            n_mels=self.cfg.num_mels,
            fmin=0,
            fmax=((self.sr / 2) + 1),
            norm=None,
        )

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def call(self, inputs):
        tensor = tf.numpy_function(self.np_call, [inputs], tf.float32, stateful=False)
        return tf.reshape(
            tensor, (len(inputs), self.cfg.num_frames, self.cfg.num_mels, 1)
        )

    def np_call(self, inputs):
        half = (self.cfg.fft_size // 2) + 1  # 512 -> 257
        if inputs.ndim == 4:
            fft_res = inputs[:, :, 0:half, 0]
        elif inputs.ndim == 3:
            fft_res = inputs[:, 0:half, 0]
        else:
            raise Exception(f"Invalid dimensions of fft_res:{fft_res.shape}")
        mag_frames = (fft_res + 1) ** 2  # 1 is added for numerical stability
        mels = np.tensordot(mag_frames, self.filter_banks.T, axes=1)
        log_mels = np.log2(mels, dtype=np.float32)
        # we floor because our approximation just gets the leading bit right
        # https://stackoverflow.com/questions/54661131/log2-approximation-in-fixed-point
        return np.expand_dims(np.floor(log_mels), axis=-1)

    def get_config(self):
        base_config = super().get_config()
        config = {"cfg": self.cfg}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(cfg=config["cfg"])
