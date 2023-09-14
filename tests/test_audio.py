import os

import numpy as np
import qkeras
import tensorflow as tf
from tensorflow.nn import softmax

from chisel4ml import generate
from chisel4ml import optimize
from chisel4ml.preprocess.fft_layer import FFTLayer

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_fft():
    frame_length = 512
    num_frames = 32
    time = np.linspace(0, 1, num_frames * frame_length)

    use_hamming_opts = [True, False]
    tone_freqs = [200, 36]
    amplitudes = [0.4, 0.8]
    functions = [np.cos, np.sin]
    for use_hamming_opt in use_hamming_opts:
        for tone_freq in tone_freqs:
            for amplitude in amplitudes:
                for function in functions:
                    # Generate test wave function
                    wave = function(2 * np.pi * tone_freq * time).reshape(32,512)
                    frames = np.round((wave + 0) * 2047 * amplitude)
                    

                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.Input(shape=(32, 512)))
                    model.add(
                        qkeras.QActivation(qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1))
                    )
                    if use_hamming_opt:
                        model.add(FFTLayer(win_fn='hamming'))
                    else:
                        model.add(FFTLayer(win_fn='none'))
                    opt_model = optimize.qkeras_model(model)
                    audio_preproc = generate.circuit(
                        opt_model=opt_model, use_verilator=True, gen_vcd=True
                    )
                    hw_res = audio_preproc(frames) / (2**4)
                    sw_res = opt_model(frames.reshape(1, 32, 512))
                    # import matplotlib.pyplot as plt
                    # plt.plot(hw_res.flatten(), color='r')
                    # plt.plot(sw_res.numpy().flatten(), color='g', linestyle='dashed')
                    # plt.show()
                    assert np.allclose(sw_res.numpy().reshape(32,512), hw_res, atol=2, rtol=0.1)


def test_preproc_speech_commands(qnn_audio_class):
    _, test_set = qnn_audio_class
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(13, 12, keep_negative=True, alpha=1))
    )
    model.add(FFTLayer())
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
        assert np.allclose(sw_res.numpy().reshape(32, 20), hw_res / (2**16), rtol=0.01)


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
