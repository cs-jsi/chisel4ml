import itertools
import os

import numpy as np
import pytest
import torch

from chisel4ml import generate
from chisel4ml.preprocess.fft_torch_op import FFTreal_layer
from chisel4ml.preprocess.lmfe_torch_op import lmfe_layer
from tests.brevitas_quantizers import Int12ActQuant
from tests.brevitas_quantizers import Int31ActQuant
from tests.brevitas_quantizers import Int32ActQuant
from tests.brevitas_quantizers import Int33ActQuant
from tests.brevitas_quantizers import UInt8ActQuant

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_frames(tone_freq, amplitude, function, frame_length, num_frames):
    time = np.linspace(0, 1, num_frames * frame_length)
    # Generate test wave function
    wave = function(2 * np.pi * tone_freq * time).reshape(num_frames, frame_length)
    frame = np.round((wave + 0) * 2047 * amplitude)
    return [(frame, 0)]  # we add this dummy 0 to be compatible with audio_data


def get_model(fft_size, num_frames, num_mels):
    output_quant_dict = {128: Int31ActQuant, 256: Int32ActQuant, 512: Int33ActQuant}

    class Model(torch.nn.Module):
        def __init__(self, fft_size, num_frames, win_fn, num_mels):
            super(Model, self).__init__()
            self.fft_size = fft_size
            self.num_frames = num_frames
            self.win_fn = torch.from_numpy(np.array(win_fn))
            self.fft = FFTreal_layer(
                fft_size=fft_size,
                num_frames=num_frames,
                win_fn=self.win_fn,
                input_quant=Int12ActQuant,
                output_quant=output_quant_dict[fft_size],
            )
            if num_mels > 0:
                self.lmfe = lmfe_layer(
                    fft_size=fft_size,
                    num_mels=num_mels,
                    num_frames=num_frames,
                    input_quant=None,
                    output_quant=UInt8ActQuant,
                )

        def forward(self, x):
            fftr = self.fft.forward(x)
            if num_mels == 0:
                return fftr
            else:
                return self.lmfe.forward(fftr)

    model = Model(
        fft_size=fft_size,
        num_frames=num_frames,
        win_fn=np.hamming(fft_size).astype(np.float32),
        num_mels=num_mels,
    )
    return model


test_opts_dict = {
    "tone_freq": (60,),
    "amplitude": (0.8,),
    "function": (np.cos,),
    "frame_length": (128, 256, 512),
    "num_frames": (8, 32),
    "data": ("generated",),
}
test_opts_list = list(itertools.product(*test_opts_dict.values()))
test_opts_list.append((0, 0.0, None, 512, 32, "audio"))


@pytest.mark.parametrize(
    "tone_freq,amplitude,function,frame_length,num_frames,data", test_opts_list
)
def test_fft(
    request, tone_freq, amplitude, function, frame_length, num_frames, data, audio_data
):
    if data == "generated":
        frames = get_frames(tone_freq, amplitude, function, frame_length, num_frames)
    elif data == "audio":
        _, _, frames, _, _, _, _ = audio_data
        frames = list(frames)[:5]
    else:
        raise ValueError

    model = get_model(
        fft_size=frame_length,
        num_frames=num_frames,
        num_mels=0,  # we dont use the lmfe layer in this test
    )
    ishape = (1, num_frames, frame_length)
    accelerators, lbir_model = generate.accelerators(
        model,
        ishape=ishape,
        minimize="area",
        debug=request.config.getoption("--debug-trans"),
    )
    audio_preproc = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
    )
    for frame, _ in frames:
        hw_res = audio_preproc(frame, sim_timeout_sec=400) / 2**12
        sw_res = model(torch.from_numpy(frame.reshape(1, num_frames, frame_length)))
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


test_opts_lmfe_dict = test_opts_dict.copy()
test_opts_lmfe_dict["num_mels"] = (10, 20)
test_opts_lmfe_list = list(itertools.product(*test_opts_lmfe_dict.values()))
test_opts_lmfe_list.append((0, 0.0, None, 512, 32, "audio", 20))


@pytest.mark.parametrize(
    "tone_freq,amplitude,function,frame_length,num_frames,data,num_mels",
    test_opts_lmfe_list,
)
def test_lmfe(
    request,
    tone_freq,
    amplitude,
    function,
    frame_length,
    num_frames,
    data,
    num_mels,
    audio_data,
):
    if data == "generated":
        frames = get_frames(tone_freq, amplitude, function, frame_length, num_frames)
    elif data == "audio":
        _, _, frames, _, _, _, _ = audio_data
        frames = list(frames)[:5]
    else:
        raise ValueError

    model = get_model(fft_size=frame_length, num_frames=num_frames, num_mels=num_mels)

    ishape = (1, num_frames, frame_length)
    accels, lbir_model = generate.accelerators(
        model,
        ishape=ishape,
        minimize="area",
        debug=request.config.getoption("--debug-trans"),
    )
    audio_preproc = generate.circuit(
        accels,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
    )
    for frame, _ in frames:
        hw_res = audio_preproc(frame, sim_timeout_sec=400) - 24
        sw_res = model(
            torch.from_numpy(
                frame.reshape(1, num_frames, frame_length).astype(np.float32)
            )
        )

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
