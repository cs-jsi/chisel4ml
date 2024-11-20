import brevitas.nn as qnn
import torch
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from tests.brevitas_quantizers import CommonWeightQuant
from tests.brevitas_quantizers import IntActQuant
from tests.brevitas_quantizers import IntBiasQuant
from tests.brevitas_quantizers import WeightPerChannelQuant
from torch import nn
from torch.nn import Module
import numpy as np

def quant_to_dt(quant):
    return DataType[f"INT{quant}"] if quant > 1 else DataType["BIPOLAR"]

def get_cnn_model(input_size, in_ch):
    in_feat = int((input_size[0] - 2) * (input_size[1] -2) / 4) * in_ch
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.ishape = (1, in_ch, input_size[0], input_size[1])
            self.quant_inp = qnn.QuantIdentity(
                bit_width=3, 
                scaling_impl_type='const', 
                scaling_init=2**(3 - 1),
                signed=True
            )
            self.conv = nn.Sequential(
                qnn.QuantConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    groups=1,
                    kernel_size=3,
                    bias=True,
                    weight_quant=WeightPerChannelQuant,
                    weight_bit_width=3,
                    bias_quant=IntBiasQuant,
                    bias_bit_width=4,
                    bias_scaling_impl_type="const",
                    bias_scaling_init=2 ** (4 - 1) - 1,
                ),
                qnn.QuantReLU(
                     bit_width=3, 
                     scaling_impl_type='const', 
                     scaling_init=(2**3) - 1
                )
            )
            self.mp = nn.MaxPool2d(
                kernel_size=2
            )
            self.quant = qnn.QuantIdentity(
                bit_width=3, 
                scaling_impl_type='const', 
                scaling_init=2**3 - 1,
                signed=False  # ReLU
            )
            self.dense = qnn.QuantLinear(
                in_features=in_feat,
                out_features=1,
                bias=True,
                weight_quant=WeightPerChannelQuant,
                weight_bit_width=3,
                bias_quant=IntBiasQuant,
                bias_bit_width=4,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (4 - 1) - 1,
            )
            self.relu = qnn.QuantReLU(
                bit_width=3, 
                scaling_impl_type='const', 
                scaling_init=(2**3) - 1
            )

        def forward(self, x):
            x = self.quant_inp(x)
            x = self.conv(x)
            x = self.mp(x)
            x = self.quant(x)
            x = torch.flatten(x)
            x = self.dense(x)
            x = self.relu(x)
            return x

    model = Model()

    # we need to initialize the mode with approprirate weights
    wshape_conv = (in_ch, in_ch , 3, 3)
    weights_conv = gen_finn_dt_tensor(quant_to_dt(3), wshape_conv)
    bias_conv = gen_finn_dt_tensor(quant_to_dt(4), (in_ch,))
    model.conv.weight = torch.nn.Parameter(torch.from_numpy(weights_conv).float())
    model.conv.bias = torch.nn.Parameter(torch.from_numpy(bias_conv).float())

    wshape_dense = (1, in_feat)
    weights_dense = gen_finn_dt_tensor(quant_to_dt(3), wshape_dense)
    bias_dense = gen_finn_dt_tensor(quant_to_dt(4), (1,))
    model.dense.weight = torch.nn.Parameter(torch.from_numpy(weights_dense).float())
    model.dense.bias = torch.nn.Parameter(torch.from_numpy(bias_dense).float())
    model.eval()

    
    test_data_shape = (8,) + model.ishape[1:]
    data = gen_finn_dt_tensor(quant_to_dt(3), test_data_shape)
    return model, data


def get_conv_layer_model(
    input_ch, output_ch, kernel_size, padding, iq, wq, bq, oq, depthwise=False
):
    class ConvLayerModel(Module):
        def __init__(self):
            super(ConvLayerModel, self).__init__()
            self.ishape = (1, input_ch, 4, 4)
            self.conv = qnn.QuantConv2d(
                in_channels=input_ch,
                out_channels=output_ch,
                groups=output_ch if depthwise else 1,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=wq,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if wq == 1 else 2 ** (wq - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=bq,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bq - 1) - 1,
                input_quant=IntActQuant,
                input_bit_width=iq,
                input_scaling_impl_type="const",
                input_scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
                output_quant=IntActQuant,
                output_bit_width=oq,
                output_scaling_impl_type="const",
                output_scaling_init=1 if oq == 1 else 2 ** (oq - 1) - 1,
            )

        def forward(self, x):
            return self.conv(x)

    model = ConvLayerModel()
    wshape = (output_ch, 1 if depthwise else input_ch, kernel_size[0], kernel_size[1])
    bshape = (output_ch,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape)
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.conv.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.conv.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data

def get_maxpool_layer_model(channels, input_size, kernel_size, padding, iq):
    class MaxPoolLayerModel(Module):
        def __init__(self, input_size):
            super(MaxPoolLayerModel, self).__init__()
            self.ishape = (1, channels) + input_size
            self.in_quant = qnn.QuantIdentity(
                act_quant=IntActQuant,
                bit_width=iq,
                scaling_impl_type="const",
                scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
            )
            self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, padding=padding)
            self.out_quant = qnn.QuantIdentity(
                act_quant=IntActQuant,
                bit_width=iq,
                scaling_impl_type="const",
                scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
            )

        def forward(self, x):
            tmp = self.in_quant(x)
            tmp = self.maxpool(tmp)
            return self.out_quant(tmp)

    model = MaxPoolLayerModel(input_size)
    # set seed for repeatability
    np.random.seed(42)
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data

def get_linear_layer_model(in_features, out_features, bias, iq, wq, bq, oq, weight_scale=1):
    class LinearLayerModel(Module):
        def __init__(self):
            wscale = 1 if wq == 1 else 2 ** (wq - 1) - 1
            wscale = wscale * weight_scale
            super(LinearLayerModel, self).__init__()
            self.ishape = (1, in_features)
            self.linear = qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                weight_quant=CommonWeightQuant,
                weight_bit_width=wq,
                weight_scaling_impl_type="const",
                weight_scaling_init=wscale,
                bias_quant=IntBiasQuant,
                bias_bit_width=bq,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bq - 1) - 1,
                input_quant=IntActQuant,
                input_bit_width=iq,
                input_scaling_impl_type="const",
                input_scaling_init=1 if iq == 1 else 2 ** (iq - 1) - 1,
                output_quant=IntActQuant,
                output_bit_width=oq,
                output_scaling_impl_type="const",
                output_scaling_init=1 if oq == 1 else 2 ** (oq - 1) - 1,
            )

        def forward(self, x):
            return self.linear(x)

    model = LinearLayerModel()
    wshape = (out_features, in_features)
    bshape = (out_features,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape) * weight_scale
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.linear.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.linear.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data