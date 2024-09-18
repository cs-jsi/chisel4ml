import onnx
from qonnx.core.modelwrapper import ModelWrapper

from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.transforms.qonnx_utils import _lmfeconfig_to_kwargs


def transform_lmfe(model: ModelWrapper, node) -> bool:
    input_qt_node = model.find_producer(node.input[0])
    output_qt_node = model.find_consumer(node.output[0])
    if input_qt_node.op_type == "QTensor":
        input_qt_str = onnx.helper.get_node_attr_value(input_qt_node, "qtensor")
        input_qt = QTensor.FromString(input_qt_str)
    elif input_qt_node.op_type == "FFTConfig":
        fft_str = onnx.helper.get_node_attr_value(input_qt_node, "fft")
        fft = FFTConfig.FromString(fft_str)
        input_qt = fft.output
    else:
        raise ValueError

    input_qt.dtype.shift[:] = [input_qt.dtype.shift[0]]
    output_qt_str = onnx.helper.get_node_attr_value(output_qt_node, "qtensor")
    output_qt = QTensor.FromString(output_qt_str)
    output_qt.dtype.shift[:] = [output_qt.dtype.shift[0]]
    filter_banks = model.get_initializer(node.input[1])
    fft_size = model.get_initializer(node.input[2]).item()
    num_frames = model.get_initializer(node.input[4]).item()
    num_mels = model.get_initializer(node.input[3]).item()
    lmfecfg = LMFEConfig(
        fft_size=fft_size,
        num_mels=num_mels,
        num_frames=num_frames,
        input=input_qt,
        output=output_qt,
        mel_filters=filter_banks.flatten().tolist(),
    )
    if input_qt_node.op_type == "QTensor":
        inputs = [input_qt_node.input[0]]
    else:
        inputs = input_qt_node.output
    kwargs = _lmfeconfig_to_kwargs(lmfecfg)
    new_node = onnx.helper.make_node(
        op_type="LMFEConfig",
        inputs=inputs,
        outputs=output_qt_node.output,
        domain="chisel4ml",
        lmfe=lmfecfg.SerializeToString(),
        **kwargs,
    )
    model.graph.node.remove(node)
    if input_qt_node.op_type == "QTensor":
        model.graph.node.remove(input_qt_node)
    model.graph.node.remove(output_qt_node)
    model.graph.node.extend([new_node])
    return True
