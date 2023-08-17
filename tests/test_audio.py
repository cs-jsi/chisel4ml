import os

import numpy as np
import qkeras
import tensorflow as tf
from tensorflow.nn import softmax

from chisel4ml import generate
from chisel4ml import optimize
from chisel4ml.preprocess.audio_preprocessing_layer import AudioPreprocessingLayer

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_preproc_sine_wave():
    tone_freq = 200
    num_frames = 32
    frame_length = 512
    sr = 32 * 512  # approx 16000

    time_axis = np.linspace(0, 1, sr)
    sine_wave = np.sin(2 * np.pi * tone_freq * time_axis)
    frames = sine_wave.reshape([num_frames, frame_length])
    frames = np.round((frames + 0) * 2047 * 0.8)

    # SW result
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(13, 12, keep_negative=True, alpha=1))
    )
    model.add(AudioPreprocessingLayer())
    opt_model = optimize.qkeras_model(model)

    audio_preproc = generate.circuit(
        opt_model=opt_model, use_verilator=True, gen_vcd=True
    )
    hw_res = audio_preproc(frames)
    sw_res = opt_model(frames.reshape(1, num_frames, frame_length))

    assert np.allclose(sw_res.numpy().reshape(32, 20), hw_res, atol=1)


def test_preproc_speech_commands(qnn_audio_class):
    _, test_set = qnn_audio_class
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(13, 12, keep_negative=True, alpha=1))
    )
    model.add(AudioPreprocessingLayer())
    opt_model = optimize.qkeras_model(model)

    audio_preproc = generate.circuit(
        opt_model=opt_model, use_verilator=True, gen_vcd=True
    )
    assert audio_preproc is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, _ = next(ts_iter)
        hw_res = audio_preproc(sample)
        sw_res = opt_model(sample.reshape(1, 32, 512))
        assert np.allclose(sw_res.numpy().reshape(32, 20), hw_res, atol=2)


def test_audio_classifier(qnn_audio_class):
    opt_model, test_set = qnn_audio_class
    circuit = generate.circuit(opt_model, use_verilator=True, gen_vcd=True)
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        hw_ret = circuit.predict(sample)
        sw_ret = opt_model.predict(sample.reshape(1, 32, 512))
        print(f"hw_ret: {np.argmax(hw_ret)} - sw_ret: {np.argmax(sw_ret)}")
        assert np.argmax(softmax(hw_ret)) == np.argmax(softmax(sw_ret))
