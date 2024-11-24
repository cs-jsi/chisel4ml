import numpy as np
from qonnx.custom_op.general.bipolar_quant import binary_quant
from qonnx.custom_op.general.quant import quant

from chisel4ml.lbir.datatype_pb2 import Datatype as LBIRDatatype
from chisel4ml.lbir.lbir_pb2 import Conv2DConfig
from chisel4ml.lbir.lbir_pb2 import DenseConfig
from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig
from chisel4ml.lbir.lbir_pb2 import MaxPool2DConfig
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


def _fftconfig_to_kwargs(layer: FFTConfig):
    kwargs = dict()
    kwargs.update(_qtensor_to_kwargs(layer.input, key_prefix="input_"))
    kwargs.update(_qtensor_to_kwargs(layer.output, key_prefix="output_"))
    kwargs["fft_size"] = layer.fft_size
    kwargs["num_frames"] = layer.num_frames
    kwargs["win_fn"] = layer.win_fn
    return kwargs


def _lmfeconfig_to_kwargs(layer: LMFEConfig):
    kwargs = dict()
    kwargs.update(_qtensor_to_kwargs(layer.input, key_prefix="input_"))
    kwargs.update(_qtensor_to_kwargs(layer.output, key_prefix="output_"))
    kwargs["fft_size"] = layer.fft_size
    kwargs["num_frames"] = layer.num_frames
    kwargs["num_mels"] = layer.num_mels
    kwargs["mel_filters"] = layer.mel_filters
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
    kwargs["depthwise"] = layer.depthwise
    kwargs["stride"] = layer.stride
    kwargs["padding"] = layer.padding
    return kwargs


def _maxpool2dconfig_to_kwargs(layer: MaxPool2DConfig):
    kwargs = dict()
    kwargs.update(_qtensor_to_kwargs(layer.input, key_prefix="input_"))
    kwargs.update(_qtensor_to_kwargs(layer.output, key_prefix="output_"))
    kwargs["kernel_shape"] = layer.kernel_shape
    kwargs["stride"] = layer.stride
    kwargs["padding"] = layer.padding
    return kwargs


def _numpy_to_bitwidth(np_arr) -> int:
    """
    The number of bits requried to represent this array. We add an
    extra bit so that same values can be represented in negative (thresh-bias)
    """
    maxval = np.abs(np_arr).max()
    if maxval < 0.0001:
        return 1
    else:
        return np.ceil(np.log2(maxval)).astype(int).item() + 2 + int(np_arr.min() < 0.0)


def _scale_to_shift(scale):
    shift = np.log2(scale).astype(int)
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


def get_lbir_shape(old_shape, old_layout, is_weight):
    # TODO: Fix layout functionality in QONNX, so that this can be a proper
    # transformation.
    if len(old_shape) == 2:
        return old_shape
    if len(old_shape) == 1:
        return (1, old_shape[0])  # batch size == 1
    elif len(old_shape) == 4:
        # We assume the layout is NCHW, because that is the default in torch
        # where conv layers are mostly produced. Unfortunately, the tensor layout
        # functionality in QONNX does not work correctly so using
        # model.get_tensor_layout to determine the actualy layout is currently not
        # possible. This can lead to problems with different layouts (like conv
        # from keras)
        if is_weight:
            return old_shape  # KCHW
        else:
            return old_shape[1:]  # CHW
    else:
        raise NotImplementedError


def qtensor_to_quantizer(qtensor):
    if qtensor.dtype.quantization == LBIRDatatype.QuantizationType.UNIFORM:
        return lambda x: quant(
            x,
            scale=np.exp2(
                np.array(qtensor.dtype.shift), dtype=np.float32
            ),  # inverse _scale_to_shift
            zeropt=np.array(qtensor.dtype.offset, dtype=np.float32),
            bitwidth=np.array(qtensor.dtype.bitwidth, dtype=np.float32),
            signed=qtensor.dtype.signed,
            narrow=False,
            rounding_mode=qtensor.rounding_mode,
        )
    else:
        return lambda x: binary_quant(x, 1.0)


def replace_tensor(tensor_list, old, new):
    "Replaces the tensor and preservers the order in the list."
    old_ind = -1
    for ind, tensor in enumerate(tensor_list):
        if tensor == old:
            old_ind = ind
            break
    if old_ind == -1:
        raise ValueError(f"Tensor {old} not in tensor list: {tensor_list}")
    tensor_list[old_ind] = new
