import brevitas.nn as qnn
import torch
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from torch import nn
from torch.nn import Module


def quant_to_dt(quantizer):
    return DataType["INT4"]


def get_brevitas_model(conv_confs, maxp_confs, dense_confs, input_size=(8, 8)):
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = []
            self.mp = []
            self.dense = []
            for cc in conv_confs:
                self.conv.append(
                    nn.Sequential(
                        qnn.QuantConv2d(
                            in_channels=cc["ich"],
                            out_channels=cc["och"],
                            groups=cc["groups"],
                            kernel_size=cc["ks"],
                            padding=cc["pa"],
                            stride=cc["st"],
                            bias=True,
                            weight_quant=cc["wq"],
                            bias_quant=cc["bq"],
                            input_quant=cc["iq"],
                            output_quant=cc["oq"],
                        ),
                        nn.ReLU(),
                    )
                )
            for mc in maxp_confs:
                self.mp.append(
                    nn.MaxPool2d(
                        kernel_size=mc["ks"], stride=mc["st"], padding=mc["pa"]
                    )
                )
            for dc in dense_confs:
                self.dense.append(
                    qnn.QuantLinear(
                        in_features=dc["in_size"],
                        out_features=dc["out_size"],
                        bias=True,
                        weight_quant=dc["wq"],
                        bias_quant=dc["bq"],
                        input_quant=dc["iq"],
                        output_quant=dc["oq"],
                    )
                )

        def forward(self, x):
            tensor = x
            for layer in self.conv + self.mp + [torch.flatten] + self.dense:
                tensor = layer(tensor)
            return tensor

    model = Model()

    # we need to initialize the mode with approprirate weights
    for ind, cc in enumerate(conv_confs):
        wshape = (cc["och"], cc["ich"] // cc["groups"], cc["ks"][0], cc["ks"][1])
        weights = gen_finn_dt_tensor(quant_to_dt(cc["wq"]), wshape)
        bias = gen_finn_dt_tensor(quant_to_dt(cc["bq"]), (cc["och"],))
        model.conv[ind].weight = torch.nn.Parameter(torch.from_numpy(weights).float())
        model.conv[ind].bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    for ind, dc in enumerate(dense_confs):
        wshape = (dc["out_size"], dc["in_size"])
        weights = gen_finn_dt_tensor(quant_to_dt(dc["wq"]), wshape)
        bias = gen_finn_dt_tensor(quant_to_dt(dc["bq"]), (dc["out_size"],))
        model.dense[ind].weight = torch.nn.Parameter(torch.from_numpy(weights).float())
        model.dense[ind].bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    model.eval()
    ishape = (1, conv_confs[0]["ich"], input_size[0], input_size[1])
    test_data_shape = (8,) + ishape[1:]
    data = gen_finn_dt_tensor(quant_to_dt(conv_confs[0]["iq"]), test_data_shape)
    return model, ishape, data
