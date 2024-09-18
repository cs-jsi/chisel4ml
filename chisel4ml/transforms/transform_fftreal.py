import onnx
from qonnx.core.modelwrapper import ModelWrapper

from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.transforms.qonnx_utils import _fftconfig_to_kwargs


def transform_fftreal(model: ModelWrapper, node) -> bool:
    input_qt_node = model.find_producer(node.input[0])
    output_qt_node = model.find_consumer(node.output[0])
    input_qt_str = onnx.helper.get_node_attr_value(input_qt_node, "qtensor")
    input_qt = QTensor.FromString(input_qt_str)
    input_qt.dtype.shift[:] = [input_qt.dtype.shift[0]]
    output_qt_str = onnx.helper.get_node_attr_value(output_qt_node, "qtensor")
    output_qt = QTensor.FromString(output_qt_str)
    output_qt.dtype.shift[:] = [output_qt.dtype.shift[0]]
    fft_size = model.get_initializer(node.input[2]).item()
    num_frames = model.get_initializer(node.input[3]).item()
    win_fn = model.get_initializer(node.input[1]).tolist()
    fftcfg = FFTConfig(
        fft_size=fft_size,
        num_frames=num_frames,
        win_fn=win_fn,
        input=input_qt,
        output=output_qt,
    )
    kwargs = _fftconfig_to_kwargs(fftcfg)
    new_node = onnx.helper.make_node(
        op_type="FFTConfig",
        inputs=[input_qt_node.input[0]],
        outputs=output_qt_node.output,
        domain="chisel4ml",
        fft=fftcfg.SerializeToString(),
        **kwargs,
    )
    model.graph.node.remove(node)
    model.graph.node.remove(input_qt_node)
    model.graph.node.remove(output_qt_node)
    model.graph.node.extend([new_node])
    return True
