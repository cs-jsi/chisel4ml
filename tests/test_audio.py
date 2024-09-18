import itertools
import os

import numpy as np
import pytest
import qkeras
import tensorflow as tf
import torch

from chisel4ml import generate
from chisel4ml import optimize
from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.fft_torch_op import FFTreal_layer
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from tests.brevitas_quantizers import Int12ActQuant
from tests.brevitas_quantizers import Int31ActQuant
from tests.brevitas_quantizers import Int32ActQuant
from tests.brevitas_quantizers import Int33ActQuant

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_fft(request):
    test_opts_dict = {
        "window": ("hamming",),
        "tone_freq": (60,),
        "amplitude": (0.8,),
        "function": (np.cos,),
        "frame_length": (128, 256, 512),
        "num_frames": (8, 32),
    }

    for test_case in itertools.product(*test_opts_dict.values()):
        tcdict = dict(zip(test_opts_dict.keys(), test_case))
        print(f"Testing FFT with options:{tcdict}")
        tone_freq = tcdict["tone_freq"]
        amplitude = tcdict["amplitude"]
        function = tcdict["function"]
        frame_length = tcdict["frame_length"]
        num_frames = tcdict["num_frames"]

        time = np.linspace(0, 1, num_frames * frame_length)
        # Generate test wave function
        wave = function(2 * np.pi * tone_freq * time).reshape(num_frames, frame_length)
        frames = np.round((wave + 0) * 2047 * amplitude)

        if frame_length == 128:
            output_quant = Int31ActQuant
        elif frame_length == 256:
            output_quant = Int32ActQuant
        elif frame_length == 512:
            output_quant = Int33ActQuant

        class Model(torch.nn.Module):
            def __init__(self, fft_size, num_frames, win_fn):
                super(Model, self).__init__()
                self.fft_size = fft_size
                self.num_frames = num_frames
                self.win_fn = torch.from_numpy(np.array(win_fn))
                self.fft = FFTreal_layer(
                    fft_size=fft_size,
                    num_frames=num_frames,
                    win_fn=self.win_fn,
                    input_quant=Int12ActQuant,
                    output_quant=output_quant,
                )

            def forward(self, x):
                return self.fft.forward(x)

        model = Model(
            fft_size=frame_length,
            num_frames=num_frames,
            win_fn=np.hamming(frame_length),
        )
        audio_preproc = generate.circuit(
            model=model,
            ishape=(1, num_frames, frame_length),
            use_verilator=request.config.getoption("--use-verilator"),
            gen_waveform=request.config.getoption("--gen-waveform"),
            waveform_type=request.config.getoption("--waveform-type"),
            gen_timeout_sec=request.config.getoption("--generation-timeout"),
            debug=request.config.getoption("--debug-trans"),
        )
        hw_res = audio_preproc(frames, sim_timeout_sec=400) / 2**12
        sw_res = model(torch.from_numpy(frames.reshape(1, num_frames, frame_length)))
        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(sw_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(
            sw_res.numpy().reshape(num_frames, frame_length),
            hw_res,
            atol=10,
            rtol=0.05,
        )
        audio_preproc.delete_from_server()


def test_fft_speech_commands(request, audio_data):
    _, _, test_set, _, _, _, _ = audio_data

    class Model(torch.nn.Module):
        def __init__(self, fft_size, num_frames, win_fn):
            super(Model, self).__init__()
            self.fft_size = fft_size
            self.num_frames = num_frames
            self.win_fn = torch.from_numpy(np.array(win_fn))
            self.fft = FFTreal_layer(
                fft_size=fft_size,
                num_frames=num_frames,
                win_fn=self.win_fn,
                input_quant=Int12ActQuant,
                output_quant=Int33ActQuant,
            )

        def forward(self, x):
            return self.fft.forward(x)

    model = Model(fft_size=512, num_frames=32, win_fn=np.hamming(512))
    audio_preproc = generate.circuit(
        model=model,
        ishape=(1, 32, 512),
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
    )
    assert audio_preproc is not None
    for sample, _ in test_set:
        hw_res = audio_preproc(sample, sim_timeout_sec=400) / 2**12
        sw_res = model(torch.from_numpy(sample.reshape(1, 32, 512)))
        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(sw_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(sw_res.numpy().reshape(32, 512), hw_res, atol=1, rtol=0.05)
    audio_preproc.delete_from_server()


@pytest.mark.skip()
def test_mel_engine(request, audio_data):
    test_opts_dict = {
        "window": ("hamming",),
        "tone_freq": (60,),
        "amplitude": (0.8,),
        "function": (np.cos,),
        "frame_length": (128, 512),
        "num_frames": (16, 32),
        "num_mels": (10, 20),
    }
    for test_case in itertools.product(*test_opts_dict.values()):
        tcdict = dict(zip(test_opts_dict.keys(), test_case))
        print(f"Testing FFT with options:{tcdict}")
        tone_freq = tcdict["tone_freq"]
        amplitude = tcdict["amplitude"]
        function = tcdict["function"]
        frame_length = tcdict["frame_length"]
        num_frames = tcdict["num_frames"]
        num_mels = tcdict["num_mels"]

        time = np.linspace(0, 1, num_frames * frame_length)
        # Generate test wave function
        wave = function(2 * np.pi * tone_freq * time).reshape(num_frames, frame_length)
        frames = np.round((wave + 0) * 2047 * amplitude)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(num_frames, frame_length)))
        model.add(
            qkeras.QActivation(
                qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1)
            )
        )
        model.add(
            FFTLayer(
                FFTConfig(
                    fft_size=frame_length,
                    num_frames=num_frames,
                    win_fn=np.hamming(frame_length),
                )
            )
        )
        model.add(
            LMFELayer(
                LMFEConfig(
                    fft_size=frame_length,
                    num_frames=num_frames,
                    num_mels=num_mels,
                )
            )
        )
        opt_model = optimize.qkeras_model(model)
        audio_preproc = generate.circuit(
            opt_model=opt_model,
            use_verilator=request.config.getoption("--use-verilator"),
            gen_waveform=request.config.getoption("--gen-waveform"),
            gen_timeout_sec=request.config.getoption("--generation-timeout"),
            debug=request.config.getoption("--debug-trans"),
        )
        hw_res = audio_preproc(frames, sim_timeout_sec=400)
        sw_res = opt_model(frames.reshape(1, num_frames, frame_length))
        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(sw_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(
            sw_res.numpy().flatten(),
            hw_res.flatten(),
            atol=10,
            rtol=0.05,
        )
        audio_preproc.delete_from_server()


@pytest.mark.skip()
def test_lmfe_speech_commands(request, audio_data):
    _, _, test_set, _, _, _, _ = audio_data
    fft_layer = FFTLayer(FFTConfig(fft_size=512, num_frames=32, win_fn=np.hamming(512)))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512, 1)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(32, 31, keep_negative=True, alpha=1))
    )
    model.add(LMFELayer(LMFEConfig(fft_size=512, num_frames=32, num_mels=20)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    opt_model = optimize.qkeras_model(model)

    audio_preproc = generate.circuit(
        opt_model=opt_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
    )
    assert audio_preproc is not None
    for sample, _ in test_set:
        fft_res = np.round(fft_layer(sample.reshape(1, 32, 512)))
        hw_res = audio_preproc(fft_res.reshape(1, 32, 512))
        sw_res = opt_model(fft_res.reshape(1, 32, 512))
        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(sw_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(sw_res.numpy().reshape(32, 20), hw_res, atol=0, rtol=0)
    audio_preproc.delete_from_server()


@pytest.mark.skip()
def test_preproc_speech_commands(request, audio_data):
    _, _, test_set, _, _, _, _ = audio_data
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 512)))
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1))
    )
    model.add(FFTLayer(FFTConfig(fft_size=512, num_frames=32, win_fn=np.hamming(512))))
    model.add(LMFELayer(LMFEConfig(fft_size=512, num_frames=32, num_mels=20)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    opt_model = optimize.qkeras_model(model)

    audio_preproc = generate.circuit(
        opt_model=opt_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        debug=request.config.getoption("--debug-trans"),
    )
    assert audio_preproc is not None
    for sample, _ in test_set:
        hw_res = audio_preproc(sample)
        sw_res = opt_model(sample.reshape(1, 32, 512))
        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(sw_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(sw_res.numpy().reshape(32, 20), hw_res, atol=1, rtol=0)
    audio_preproc.delete_from_server()
