from typing import Optional
from typing import Union

import torch
from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantInputOutputLayer
from brevitas.quant_tensor import QuantTensor
from torch import Tensor
from torch._custom_op import impl as custom_op
from torch.nn import Module


@custom_op.custom_op("chisel4ml::FFTreal")
def FFTreal(
    x: torch.Tensor, win_fn: torch.Tensor, fft_size: int, num_frames: int
) -> torch.Tensor:
    return torch.fft.fft(x * win_fn, norm="backward").real


@FFTreal.impl("cpu")
def FFTreal_cpu(
    x: torch.Tensor, win_fn: torch.Tensor, fft_size: int, num_frames: int
) -> torch.Tensor:
    return torch.fft.fft(x * win_fn, norm="backward").real


@FFTreal.impl_abstract()
def FFTreal_abstract(x, win_fn, fft_size, num_frames):
    assert len(x.shape) == 3
    batch_x, num_frames_x, frame_length_x = x.shape
    assert num_frames_x == num_frames
    assert frame_length_x == fft_size
    assert len(win_fn).shape == 1
    assert len(win_fn) == fft_size
    return torch.empty((batch_x, num_frames, fft_size))


def FFTreal_symbolic(g, x, win_fn, fft_size, num_frames):
    return g.op("chisel4ml.preprocess::FFTreal", x, win_fn, fft_size, num_frames)


torch.onnx.register_custom_op_symbolic("chisel4ml::FFTreal", FFTreal_symbolic, 1)


class FFTreal_layer(QuantInputOutputLayer, Module):
    def __init__(
        self,
        fft_size: int,
        num_frames: int,
        win_fn,
        input_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
        output_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
        tie_input_output_quant=False,
        return_quant_tensor: bool = False,
        **kwargs,
    ) -> None:
        Module.__init__(self)
        QuantInputOutputLayer.__init__(
            self,
            input_quant,
            output_quant,
            tie_input_output_quant,
            return_quant_tensor,
            **kwargs,
        )
        self.fft_size = fft_size
        self.num_frames = num_frames
        self.win_fn = win_fn

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        input = self.unpack_input(input)
        if self.export_mode:
            assert self.cache_quant_io_metadata_only, "Can't cache multiple inputs"
            out = self.export_handler(inp=input.value)
            self._set_global_is_quant_layer(False)
            return out
        quant_input = self.input_quant(input)
        output = FFTreal(
            quant_input.value,
            win_fn=self.win_fn,
            fft_size=self.fft_size,
            num_frames=self.num_frames,
        )
        quant_output = self.output_quant(output)
        return self.pack_output(quant_output)
