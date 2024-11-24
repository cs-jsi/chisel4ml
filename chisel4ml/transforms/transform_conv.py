import logging

import numpy as np
import onnx
from qonnx.core.modelwrapper import ModelWrapper

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.lbir.lbir_pb2 import Conv2DConfig
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.transforms.qonnx_utils import _conv2dconfig_to_kwargs
from chisel4ml.transforms.transform_matmul import node_has_attr


def transform_conv(model: ModelWrapper, node) -> bool:
    """
    First checks that the Conv node has two QTensors as inputs and is followed by
    an Add node and an (optional) activation (ReLU) node.
    """
    model_changed = False
    b_err_str = (
        f"Could not transform node: {node.op_type}, with inputs: {node.input},"
        f" and outputs: {node.output}."
    )
    if len(node.input) != 2:
        logging.warning(f"{b_err_str} Because it does not have exactly 2 inputs.")
        return False
    input_node = model.find_producer(node.input[0])
    weights_node = model.find_producer(node.input[1])
    if not node_has_attr(weights_node, "values"):
        logging.warning(f"{b_err_str} Because weight node has no values")
        return False
    if input_node.op_type == "QTensor":
        input_qtensor = QTensor.FromString(
            onnx.helper.get_node_attr_value(input_node, "qtensor")
        )
    else:
        raise ValueError(f"Input node should be QTensor, not {input_node.op_type}")
    suc = model.find_direct_successors(node)
    if len(suc) != 1:
        logging.warning(f"{b_err_str} Because it does not have exactly 1 output.")
        return False
    if suc[0].op_type != "Add":
        logging.warning(
            f"{b_err_str} Because successor of matmul is not Add, but {suc[0].op_type}."
        )
        return False
    if len(suc[0].input) != 2:
        logging.warning(
            f"{b_err_str} Because Add node has {len(suc[0].input)} instead of 2 inputs."
        )
        return False
    add_node = suc[0]
    temp_inp_0 = model.find_producer(add_node.input[0])
    temp_inp_1 = model.find_producer(add_node.input[1])
    bias_node = temp_inp_0 if temp_inp_0.op_type == "QTensor" else temp_inp_1
    sucsuc = model.find_direct_successors(add_node)
    assert sucsuc is not None
    if len(sucsuc) != 1:
        logging.warning(
            f"{b_err_str} Successor of matmul has {len(sucsuc)} succesors, not 1."
        )
        return False
    if not sucsuc[0].op_type in ("Relu", "QTensor"):
        logging.warning(
            (
                f"{b_err_str} Because successor succesor is neither an activation nor"
                f"QTensor, but {sucsuc.op_type}."
            )
        )
        return False
    if sucsuc[0].op_type in ("Relu"):
        activation = lbir.Activation.RELU
        sucsucsuc = model.find_direct_successors(sucsuc[0])
        if len(sucsucsuc) != 1:
            logging.warning(
                f"{b_err_str} Successor of activation node has no output node."
            )
            return False
        if sucsucsuc[0].op_type != "QTensor":
            logging.warning(
                f"{b_err_str} Successor of activation node has no output qtensor."
            )
            return False
        output_node = sucsucsuc[0]
        activation_node = sucsuc[0]
    else:
        assert sucsuc[0].op_type == "QTensor"
        output_node = sucsuc[0]
        activation_node = None
        if onnx.helper.get_node_attr_value(output_node, "quantization") == b"BINARY":
            activation = lbir.Activation.BINARY_SIGN
        else:
            activation = lbir.Activation.NO_ACTIVATION

    thresh = QTensor.FromString(onnx.helper.get_node_attr_value(bias_node, "qtensor"))
    new_vals = (-np.array(thresh.values)).tolist()  # Threshold is oposite of bias
    del thresh.values[:]
    thresh.values.extend(new_vals)
    groups = onnx.helper.get_node_attr_value(node, "group")
    channels = model.get_tensor_shape(node.input[0])[1]
    conv2dcfg = Conv2DConfig(
        thresh=thresh,
        kernel=QTensor.FromString(
            onnx.helper.get_node_attr_value(weights_node, "qtensor")
        ),
        input=input_qtensor,
        output=QTensor.FromString(
            onnx.helper.get_node_attr_value(output_node, "qtensor")
        ),
        activation=activation,
        depthwise=(groups == channels),
        stride=onnx.helper.get_node_attr_value(node, "strides"),
        padding=onnx.helper.get_node_attr_value(node, "pads"),
    )
    for n in (node, weights_node, bias_node, add_node):
        model.graph.node.remove(n)
    if activation_node is not None:
        model.graph.node.remove(activation_node)
    kwargs = _conv2dconfig_to_kwargs(conv2dcfg)
    new_node = onnx.helper.make_node(
        op_type="QConv",
        inputs=node.input,
        outputs=output_node.input,
        domain="chisel4ml",
        qconv=conv2dcfg.SerializeToString(),
        **kwargs,
    )
    model.graph.node.extend([new_node])
    model_changed = True
    return model_changed
