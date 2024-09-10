import brevitas.nn as qnn
import torch
from pytest_cases import case
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from torch.nn import Module

from tests.brevitas_quantizers import Int4ActQuant
from tests.brevitas_quantizers import Int8BiasQuant
from tests.brevitas_quantizers import LearnedSFInt4WeightPerChannel

torch.manual_seed(0)  # set a seed to make sure the random weight init is reproducible


@case(tags="brevitas")
def case_sint_conv_layer_brev():
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    w_shape = (out_channels, in_channels, kernel_size, kernel_size)
    ishape = (1, in_channels, 4, 4)

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=1,
                kernel_size=kernel_size,
                padding=0,
                stride=1,
                bias=True,
                weight_quant=LearnedSFInt4WeightPerChannel,
                bias_quant=Int8BiasQuant,
                input_quant=Int4ActQuant,
                output_quant=Int4ActQuant,
            )

        def forward(self, x):
            return self.conv(x)

    model = Model()
    weight_tensor = gen_finn_dt_tensor(DataType["INT4"], w_shape)
    bias_tensor = gen_finn_dt_tensor(DataType["INT8"], (out_channels,))
    model.conv.weight = torch.nn.Parameter(torch.from_numpy(weight_tensor).float())
    model.conv.bias = torch.nn.Parameter(torch.from_numpy(bias_tensor).float())
    model.eval()
    test_data_shape = (8,) + ishape[
        1:
    ]  # "increase" the batch dim so there are more tests
    data = gen_finn_dt_tensor(DataType["INT4"], test_data_shape)
    return model, ishape, data
