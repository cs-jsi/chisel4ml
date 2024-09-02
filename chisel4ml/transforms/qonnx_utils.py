import numpy as np

from chisel4ml.lbir.datatype_pb2 import Datatype as LBIRDatatype
from chisel4ml.lbir.lbir_pb2 import Conv2DConfig
from chisel4ml.lbir.lbir_pb2 import DenseConfig
from chisel4ml.lbir.qtensor_pb2 import QTensor

_quant_to_string_dict = {0: "UNIFORM", 1: "BINARY"}
_act_to_string_dict = {0: "BINARY_SIGN", 1: "RELU", 2: "NO_ACTIVATION"}


def _qtensor_to_kwargs(qtensor: QTensor, key_prefix=""):
    kwargs = dict()
    kwargs[f"{key_prefix}quantization"] = _quant_to_string_dict[
        qtensor.dtype.quantization
    ]
    kwargs[f"{key_prefix}signed"] = qtensor.dtype.signed
    kwargs[f"{key_prefix}bitwidth"] = qtensor.dtype.bitwidth
    kwargs[f"{key_prefix}shift"] = qtensor.dtype.shift
    kwargs[f"{key_prefix}offset"] = qtensor.dtype.offset
    kwargs[f"{key_prefix}shape"] = qtensor.shape
    if len(qtensor.values) > 0:
        kwargs[f"{key_prefix}values"] = qtensor.values
    if qtensor.rounding_mode != "":
        kwargs[f"{key_prefix}rounding_mode"] = qtensor.rounding_mode
    return kwargs


def _denseconfig_to_kwargs(layer: DenseConfig):
    kwargs = dict()
    kwargs.update(_qtensor_to_kwargs(layer.input, key_prefix="input_"))
    kwargs.update(_qtensor_to_kwargs(layer.output, key_prefix="output_"))
    kwargs.update(_qtensor_to_kwargs(layer.thresh, key_prefix="thresh_"))
    kwargs.update(_qtensor_to_kwargs(layer.kernel, key_prefix="kernel_"))
    kwargs["activation"] = _act_to_string_dict[layer.activation]
    return kwargs


def _conv2dconfig_to_kwargs(layer: Conv2DConfig):
    kwargs = dict()
    kwargs.update(_qtensor_to_kwargs(layer.input, key_prefix="input_"))
    kwargs.update(_qtensor_to_kwargs(layer.output, key_prefix="output_"))
    kwargs.update(_qtensor_to_kwargs(layer.thresh, key_prefix="thresh_"))
    kwargs.update(_qtensor_to_kwargs(layer.kernel, key_prefix="kernel_"))
    kwargs["activation"] = _act_to_string_dict[layer.activation]
    return kwargs


def _numpy_to_bitwidth(np_arr) -> int:
    "The number of bits requried to represent this array."
    # TODO: This is not completely correct
    maxval = np.abs(np_arr).max()
    return np.ceil(np.log2(maxval)).astype(int).item() + 1


def _scale_to_shift(scale, num_nodes):
    shift = np.log2(scale).astype(int)
    if shift.size == 1:
        return shift.flatten().tolist() * num_nodes
    else:
        return shift.flatten().tolist()


def _numpy_to_qtensor(np_arr) -> QTensor:
    "Tries to convert a numpy array to a QTensor with minimal quantization settings."
    if not np.array_equal(np_arr, np_arr.astype(int)):
        raise ValueError
    qt = QTensor(
        dtype=LBIRDatatype(
            quantization=LBIRDatatype.QuantizationType.UNIFORM,
            signed=np_arr.min() < 0.0,
            bitwidth=_numpy_to_bitwidth(np_arr),
            shift=[0],
            offset=[0],
        ),
        shape=np_arr.shape,
        values=np_arr.flatten().tolist(),
    )
    return qt
