from typing import Optional
from typing import Union

import librosa
import torch
from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantInputOutputLayer
from brevitas.quant_tensor import QuantTensor
from torch import Tensor
from torch._custom_op import impl as custom_op
from torch.nn import Module


@custom_op.custom_op("chisel4ml::lmfe")
def lmfe(
    x: torch.Tensor,
    filter_banks: torch.Tensor,
    fft_size: int,
    num_mels: int,
    num_frames: int,
) -> torch.Tensor:
    half = (fft_size // 2) + 1
    fft_res = x[:, :, 0:half]
    mag_frames = (fft_res + 1) ** 2  # 1 is added for numerical stability
    mels = mag_frames @ filter_banks.T
    log_mels = torch.log2(mels)
    return log_mels


@lmfe.impl("cpu")
def lmfe_cpu(
    x: torch.Tensor,
    filter_banks: torch.Tensor,
    fft_size: int,
    num_mels: int,
    num_frames: int,
) -> torch.Tensor:
    half = (fft_size // 2) + 1
    fft_res = x[:, :, 0:half]
    mag_frames = (fft_res + 1) ** 2  # 1 is added for numerical stability
    mels = mag_frames @ filter_banks.T
    log_mels = torch.log2(mels)
    return torch.floor(log_mels)


@lmfe.impl_abstract()
def lmfe_abstract(x, filter_banks, fft_size, num_mels, num_frames):
    assert len(x.shape) == 3
    batch_x, num_frames_x, frame_length_x = x.shape
    assert num_frames_x == num_frames
    assert frame_length_x == fft_size
    return torch.empty((batch_x, num_frames, num_mels))


def lmfe_symbolic(g, x, filter_banks, fft_size, num_mels, num_frames):
    return g.op(
        "chisel4ml.preprocess::lmfe", x, filter_banks, fft_size, num_mels, num_frames
    )


torch.onnx.register_custom_op_symbolic("chisel4ml::lmfe", lmfe_symbolic, 1)


class lmfe_layer(QuantInputOutputLayer, Module):
    def __init__(
        self,
        fft_size: int,
        num_mels: int,
        num_frames: int,
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
        self.num_mels = num_mels
        self.num_frames = num_frames
        self.sr = self.num_frames * self.fft_size
        self.filter_banks = torch.from_numpy(
            librosa.filters.mel(
                n_fft=self.fft_size,
                sr=self.sr,
                n_mels=self.num_mels,
                fmin=0,
                fmax=((self.sr / 2) + 1),
                norm=None,
            )
        )

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        input = self.unpack_input(input)
        if self.export_mode:
            assert self.cache_quant_io_metadata_only, "Can't cache multiple inputs"
            out = self.export_handler(inp=input.value)
            self._set_global_is_quant_layer(False)
            return out
        quant_input = self.input_quant(input)

        output = lmfe(
            quant_input if isinstance(quant_input, Tensor) else quant_input.value,
            filter_banks=self.filter_banks,
            fft_size=self.fft_size,
            num_mels=self.num_mels,
            num_frames=self.num_frames,
        )
        quant_output = self.output_quant(output)
        return self.pack_output(quant_output)
