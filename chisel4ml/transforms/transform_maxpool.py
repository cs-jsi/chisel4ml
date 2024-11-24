import logging

import onnx
from qonnx.core.modelwrapper import ModelWrapper

from chisel4ml.lbir.lbir_pb2 import MaxPool2DConfig
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.transforms.qonnx_utils import _maxpool2dconfig_to_kwargs


def transform_maxpool(model: ModelWrapper, node) -> bool:
    """
    First checks that the MaxPool node has one QTensor as input and one as output.
    Then transforms onnx.MaxPool to lbir.MaxPoolConfig
    """
    model_changed = False
    b_err_str = (
        f"Could not transform node: {node.op_type}, with inputs: {node.input},"
        f" and outputs: {node.output}."
    )
    if len(node.input) != 1:
        logging.warning(f"{b_err_str} Because it does not have exactly 1 input.")
        return False
    input_node = model.find_producer(node.input[0])
    if input_node.op_type == "QTensor":
        input_qtensor = QTensor.FromString(
            onnx.helper.get_node_attr_value(input_node, "qtensor")
        )
    else:
        logging.warning(
            f"{b_err_str} Because it does not have a qtensor input,"
            f" but {input_node.op_type}."
        )
        return False
    output_nodes = model.find_consumers(node.output[0])
    if len(output_nodes) != 1:
        logging.warning(f"{b_err_str} Because it does not have exactly 1 output.")
        return False
    if output_nodes[0].op_type == "QTensor":
        output_qtensor = QTensor.FromString(
            onnx.helper.get_node_attr_value(output_nodes[0], "qtensor")
        )
    else:
        logging.warning(
            f"{b_err_str} Because successor of MaxPool is not QTensor,"
            f" but {output_nodes[0].op_type}."
        )
        return False
    maxpool2dcfg = MaxPool2DConfig(
        input=input_qtensor,
        output=output_qtensor,
        kernel_shape=onnx.helper.get_node_attr_value(node, "kernel_shape"),
        stride=onnx.helper.get_node_attr_value(node, "strides"),
        padding=onnx.helper.get_node_attr_value(node, "pads"),
    )
    model.graph.node.remove(node)
    kwargs = _maxpool2dconfig_to_kwargs(maxpool2dcfg)
    new_node = onnx.helper.make_node(
        op_type="MaxPool2D",
        inputs=node.input,
        outputs=node.output,
        domain="chisel4ml",
        maxpool2d=maxpool2dcfg.SerializeToString(),
        **kwargs,
    )
    model.graph.node.extend([new_node])
    model_changed = True
    return model_changed
