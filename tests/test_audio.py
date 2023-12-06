import os

import numpy as np
import qkeras
import tensorflow as tf
from tensorflow.nn import softmax

from chisel4ml import generate
from chisel4ml import optimize
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_fft():
    frame_length = 512
    num_frames = 32
    time = np.linspace(0, 1, num_frames * frame_length)

    windows = ["hamming", "none"]
    tone_freqs = [200, 36]
    amplitudes = [0.4, 0.8]
    functions = [np.cos, np.sin]
    for win in windows:
        for tone_freq in tone_freqs:
            for amplitude in amplitudes:
                for function in functions:
                    # Generate test wave function
                    wave = function(2 * np.pi * tone_freq * time).reshape(32, 512)
                    frames = np.round((wave + 0) * 2047 * amplitude)

                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.Input(shape=(32, 512)))
                    model.add(
                        qkeras.QActivation(
                            qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1)
                        )
                    )
                    model.add(FFTLayer(win_fn=win))
                    opt_model = optimize.qkeras_model(model)
                    audio_preproc = generate.circuit(
                        opt_model=opt_model, use_verilator=True, gen_waveform=True
                    )
                    hw_res = audio_preproc(frames) / (2**12)
                    sw_res = opt_model(frames.reshape(1, 32, 512))
                    # import matplotlib.pyplot as plt
                    # plt.plot(hw_res.flatten(), color='r')
                    # plt.plot(sw_res.numpy().flatten(), color='g', linestyle='dashed')
                    # plt.show()
                    assert np.allclose(
                        sw_res.numpy().reshape(32, 512), hw_res, atol=1, rtol=0.05
                    )


def test_fft_speech_commands(audio_data):
    _, _, test_set, _, _, _, _ = audio_data
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1))
    )
    model.add(FFTLayer(win_fn="hamming"))
    opt_model = optimize.qkeras_model(model)

    audio_preproc = generate.circuit(
        opt_model=opt_model, use_verilator=True, gen_waveform=True
    )
    assert audio_preproc is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(10):
        sample, _ = next(ts_iter)
        hw_res = audio_preproc(sample) / (2**12)
        sw_res = opt_model(sample.reshape(1, 32, 512))
        # import matplotlib.pyplot as plt
        # plt.plot(hw_res.flatten(), color='r')
        # plt.plot(sw_res.numpy().flatten(), color='g', linestyle='dashed')
        # plt.show()
        assert np.allclose(sw_res.numpy().reshape(32, 512), hw_res, atol=1, rtol=0.05)


def test_preproc_speech_commands(audio_data):
    _, _, test_set, _, _, _, _ = audio_data
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1))
    )
    model.add(FFTLayer(win_fn="hamming"))
    model.add(LMFELayer())
    opt_model = optimize.qkeras_model(model)

    audio_preproc = generate.circuit(
        opt_model=opt_model, use_verilator=True, gen_waveform=True
    )
    assert audio_preproc is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(20):
        sample, _ = next(ts_iter)
        hw_res = audio_preproc(sample)
        sw_res = opt_model(sample.reshape(1, 32, 512))
        # import matplotlib.pyplot as plt
        # plt.plot(hw_res.flatten(), color='r')
        # plt.plot(sw_res.numpy().flatten(), color='g', linestyle='dashed')
        # plt.show()
        assert np.allclose(sw_res.numpy().reshape(32, 20), hw_res, atol=5, rtol=0)


def test_audio_classifier_no_preproc_no_bias_1st_layer(
    qnn_audio_class_no_preproc_no_bias, audio_data_preproc
):
    _, _, test_set, _, _, _, _ = audio_data_preproc
    opt_model = qnn_audio_class_no_preproc_no_bias
    circuit = generate.circuit(
        opt_model, use_verilator=True, gen_waveform=True, num_layers=1
    )
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        sw_ret = opt_model.layers[2](opt_model.layers[1](sample.reshape(1, 32, 20, 1)))
        sw_ret = np.moveaxis(sw_ret.numpy().reshape(30, 18, 1), -1, 0)
        hw_ret = circuit.predict(sample.reshape(1, 32, 20))
        assert np.array_equal(hw_ret, sw_ret)


def test_audio_classifier_no_preproc_no_bias_1st_2nd_layer(
    qnn_audio_class_no_preproc_no_bias, audio_data_preproc
):
    _, _, test_set, _, _, _, _ = audio_data_preproc
    opt_model = qnn_audio_class_no_preproc_no_bias
    circuit = generate.circuit(
        opt_model, use_verilator=True, gen_waveform=True, num_layers=2
    )
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(500):
        sample, label = next(ts_iter)
        sw_ret = opt_model.layers[2](opt_model.layers[1](sample.reshape(1, 32, 20, 1)))
        sw_ret = opt_model.layers[4](opt_model.layers[3](sw_ret))
        sw_ret = np.moveaxis(sw_ret.numpy().reshape(28, 16, 2), -1, 0)
        hw_ret = circuit.predict(sample.reshape(1, 32, 20))
        assert np.array_equal(hw_ret, sw_ret)


def test_audio_classifier_no_preproc_no_bias_1st_2nd_3rd_layer(
    qnn_audio_class_no_preproc_no_bias, audio_data_preproc
):
    _, _, test_set, _, _, _, _ = audio_data_preproc
    opt_model = qnn_audio_class_no_preproc_no_bias
    circuit = generate.circuit(
        opt_model, use_verilator=True, gen_waveform=True, num_layers=3
    )
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        sw_ret = opt_model.layers[2](opt_model.layers[1](sample.reshape(1, 32, 20, 1)))
        sw_ret = opt_model.layers[4](opt_model.layers[3](sw_ret))
        sw_ret = opt_model.layers[5](sw_ret)
        sw_ret = np.moveaxis(sw_ret.numpy().reshape(14, 8, 2), -1, 0)
        hw_ret = circuit.predict(sample.reshape(1, 32, 20))
        assert np.array_equal(hw_ret, sw_ret)


def test_audio_classifier_no_preproc_no_bias(
    qnn_audio_class_no_preproc_no_bias, audio_data_preproc
):
    _, _, test_set, _, _, _, _ = audio_data_preproc
    opt_model = qnn_audio_class_no_preproc_no_bias
    circuit = generate.circuit(opt_model, use_verilator=True, gen_waveform=True)
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        hw_ret = circuit.predict(sample.reshape(1, 32, 20))
        sw_ret = opt_model.predict(sample.reshape(1, 32, 20, 1))
        print(f"hw_ret: {np.argmax(hw_ret)} - sw_ret: {np.argmax(sw_ret)}")
        assert np.array_equal(hw_ret, sw_ret)


def test_audio_classifier_no_preproc_1st_layer(
    qnn_audio_class_no_preproc, audio_data_preproc
):
    _, _, test_set, _, _, _, _ = audio_data_preproc
    opt_model = qnn_audio_class_no_preproc
    circuit = generate.circuit(
        opt_model, use_verilator=True, gen_waveform=True, num_layers=1
    )
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        sw_ret = opt_model.layers[3](opt_model.layers[2](sample.reshape(1, 32, 20, 1)))
        sw_ret = np.moveaxis(sw_ret.numpy().reshape(30, 18, 1), -1, 0)
        hw_ret = circuit.predict(sample.reshape(1, 32, 20))
        assert np.allclose(hw_ret, sw_ret, atol=1.0, rtol=0.0)


def test_audio_classifier_no_preproc(qnn_audio_class_no_preproc, audio_data_preproc):
    _, _, test_set, _, _, _, _ = audio_data_preproc
    opt_model = qnn_audio_class_no_preproc
    circuit = generate.circuit(opt_model, use_verilator=True, gen_waveform=True)
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        hw_ret = circuit.predict(sample.reshape(1, 32, 20))
        sw_ret = opt_model.predict(sample.reshape(1, 32, 20, 1))
        print(f"hw_ret: {np.argmax(hw_ret)} - sw_ret: {np.argmax(sw_ret)}")
        assert np.argmax(softmax(hw_ret)) == np.argmax(softmax(sw_ret))


def test_audio_classifier_full(qnn_audio_class, audio_data):
    _, _, test_set, _, _, _, _ = audio_data
    opt_model = qnn_audio_class
    circuit = generate.circuit(opt_model, use_verilator=True, gen_waveform=True)
    assert circuit is not None
    ts_iter = test_set.as_numpy_iterator()
    for _ in range(100):
        sample, label = next(ts_iter)
        hw_ret = circuit.predict(sample)
        sw_ret = opt_model.predict(sample.reshape(1, 32, 512))
        print(f"hw_ret: {np.argmax(hw_ret)} - sw_ret: {np.argmax(sw_ret)}")
        assert np.argmax(softmax(hw_ret)) == np.argmax(softmax(sw_ret))
