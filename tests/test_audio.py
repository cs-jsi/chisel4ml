import itertools
import os

import numpy as np
import pytest
import torch
from qonnx.core.onnx_exec import execute_onnx

from chisel4ml import generate
from chisel4ml import transform
from chisel4ml.preprocess.fft_torch_op import FFTreal_layer
from chisel4ml.preprocess.lmfe_torch_op import lmfe_layer
from tests.brevitas_quantizers import Int12ActQuant
from tests.brevitas_quantizers import Int31ActQuant
from tests.brevitas_quantizers import Int32ActQuant
from tests.brevitas_quantizers import Int33ActQuant
from tests.brevitas_quantizers import UInt8ActQuant
from tests.test_services import _brevitas_to_qonnx

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_frames(tone_freq, amplitude, function, frame_length, num_frames):
    time = np.linspace(0, 1, num_frames * frame_length)
    # Generate test wave function
    wave = function(2 * np.pi * tone_freq * time).reshape(num_frames, frame_length)
    frame = np.round((wave + 0) * 2047 * amplitude)
    return [(frame, 0)]  # we add this dummy 0 to be compatible with audio_data


def get_brevitas_model(fft_size, num_frames, num_mels):
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
}
test_opts_list = list(itertools.product(*test_opts_dict.values()))


@pytest.mark.skip("QONNX CustomOp behavior needs to be updated.")
@pytest.mark.parametrize(
    "tone_freq,amplitude,function,frame_length,num_frames", test_opts_list
)
def test_fft(
    request,
    c4ml_server,
    tone_freq,
    amplitude,
    function,
    frame_length,
    num_frames,
):
    frames = get_frames(tone_freq, amplitude, function, frame_length, num_frames)
    brevitas_model = get_brevitas_model(
        fft_size=frame_length,
        num_frames=num_frames,
        num_mels=0,  # we dont use the lmfe layer in this test
    )
    ishape = (1, num_frames, frame_length)
    qonnx_model = _brevitas_to_qonnx(brevitas_model, ishape)
    lbir_model = transform.qonnx_to_lbir(
        qonnx_model,
        debug=request.config.getoption("--debug-trans"),
    )
    accelerators = generate.accelerators(
        lbir_model,
        minimize="area",
    )
    audio_preproc = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server,
    )
    for frame, _ in frames:
        expanded_x = frame.reshape(1, num_frames, frame_length)
        input_name = qonnx_model.model.graph.input[0].name
        qonnx_res = execute_onnx(qonnx_model, {input_name: expanded_x})
        qonnx_res = qonnx_res[list(qonnx_res.keys())[0]]
        hw_res = audio_preproc(frame, sim_timeout_sec=400) / 2**12
        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(qonnx_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(
            qonnx_res.reshape(num_frames, frame_length),
            hw_res,
            atol=10,
            rtol=0.05,
        )
    audio_preproc.delete_from_server()


test_opts_lmfe_dict = test_opts_dict.copy()
test_opts_lmfe_dict["num_mels"] = (10, 20)
test_opts_lmfe_list = list(itertools.product(*test_opts_lmfe_dict.values()))


@pytest.mark.skip(
    "One test fails. Should be revisited when generalizing scaling factor behavior."
)
@pytest.mark.parametrize(
    "tone_freq,amplitude,function,frame_length,num_frames,num_mels",
    test_opts_lmfe_list,
)
def test_lmfe(
    request,
    c4ml_server,
    tone_freq,
    amplitude,
    function,
    frame_length,
    num_frames,
    num_mels,
):
    frames = get_frames(tone_freq, amplitude, function, frame_length, num_frames)
    brevitas_model = get_brevitas_model(
        fft_size=frame_length, num_frames=num_frames, num_mels=num_mels
    )
    ishape = (1, num_frames, frame_length)
    qonnx_model = _brevitas_to_qonnx(brevitas_model, ishape)
    lbir_model = transform.qonnx_to_lbir(
        qonnx_model,
        debug=request.config.getoption("--debug-trans"),
    )
    accels = generate.accelerators(
        lbir_model,
        minimize="area",
    )
    audio_preproc = generate.circuit(
        accels,
        lbir_model,
        use_verilator=request.config.getoption("--use-verilator"),
        gen_waveform=request.config.getoption("--gen-waveform"),
        waveform_type=request.config.getoption("--waveform-type"),
        gen_timeout_sec=request.config.getoption("--generation-timeout"),
        server=c4ml_server,
    )
    for frame, _ in frames:
        hw_res = audio_preproc(frame, sim_timeout_sec=400) - 24
        expanded_x = frame.reshape(1, num_frames, frame_length)
        input_name = qonnx_model.model.graph.input[0].name
        qonnx_res = execute_onnx(qonnx_model, {input_name: expanded_x})
        qonnx_res = qonnx_res[list(qonnx_res.keys())[0]]

        if request.config.getoption("--visualize"):
            import matplotlib.pyplot as plt

            plt.plot(hw_res.flatten(), color="r")
            plt.plot(qonnx_res.numpy().flatten(), color="g", linestyle="dashed")
            plt.show()
        assert np.allclose(
            qonnx_res.flatten(),
            hw_res.flatten(),
            atol=10,
            rtol=0.05,
        )
    audio_preproc.delete_from_server()
