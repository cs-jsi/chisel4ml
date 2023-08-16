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

log = logging.getLogger(__name__)


class AudioPreprocessingLayer(tf.keras.layers.Layer):
    """TODO"""

    def __init__(self):
        super(AudioPreprocessingLayer, self).__init__()
        self.n_mels = 20
        self.num_frames = 32
        self.frame_length = 512
        self.sr = self.num_frames * self.frame_length  # approx 16000
        self.filter_banks = librosa.filters.mel(
            n_fft=self.frame_length,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=0,
            fmax=((self.sr / 2) + 1),
            norm=None,
        )
        self.hw = np.hamming(self.frame_length)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 32, 512), dtype=tf.float32)]
    )
    def call(self, inputs):
        tensor = tf.numpy_function(self.np_call, [inputs], tf.float32, stateful=False)
        tensor.set_shape(tf.TensorShape([inputs.shape[0], 32, 20, 1]))
        return tensor

    def np_call(self, inputs):
        fft_res = np.fft.rfft(inputs * self.hw, norm="forward")
        mag_frames = fft_res.real**2
        mels = np.tensordot(self.filter_banks, mag_frames.T, axes=1)
        mels = np.where(mels == 0, np.finfo(float).eps, mels)  # Numerical stability
        log_mels = np.log2(mels, dtype=np.float32)
        return np.expand_dims(
            np.floor(log_mels).T, axis=-1
        )  # Transpose to be equivalent to hw implementation

    def get_config(self):
        base_config = super().get_config()
        config = {}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls()
