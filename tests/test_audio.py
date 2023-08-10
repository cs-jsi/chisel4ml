import os

import librosa
import numpy as np
import tensorflow as tf

from chisel4ml import generate

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_preproc_sine_wave():
    tone_freq = 200
    num_frames = 32
    frame_length = 512
    sr = 32 * 512  # approx 16000

    time_axis = np.linspace(0, 1, sr)
    sine_wave = np.sin(2 * np.pi * tone_freq * time_axis)
    frames = sine_wave.reshape([num_frames, frame_length])
    frames = np.round((frames + 1) * 2047 * 0.8)

    # SW result
    filter_banks = librosa.filters.mel(
        n_fft=frame_length, sr=sr, n_mels=20, fmin=0, fmax=((sr / 2) + 1), norm=None
    )
    hw = np.hamming(512)
    fft_res = np.fft.rfft(frames * hw, norm="forward")
    mag_frames = fft_res.real**2
    mels = np.dot(filter_banks, mag_frames.T)
    log_mels = np.log2(mels, dtype=np.float32)
    sw_res = np.floor(log_mels)

    audio_preproc = generate.circuit(
        opt_model=tf.keras.Model(), get_mfcc=True, use_verilator=True, gen_vcd=True
    )
    ret = audio_preproc(frames)
    hw_res = ret.transpose()

    assert sw_res.shape == (20, 32)
    assert ret.shape == (32, 20)
    assert np.allclose(sw_res, hw_res, atol=2)


def test_audio_classifier(qnn_audio_class):
    opt_model, test_set, test_set_no_preproc = qnn_audio_class
    circuit = generate.circuit(
        opt_model, get_mfcc=True, use_verilator=True, gen_vcd=True
    )
    titer = test_set_no_preproc.as_numpy_iterator()
    sample, label = next(titer)
    circuit.predict(sample.reshape(32, 512))
    assert circuit is not None
