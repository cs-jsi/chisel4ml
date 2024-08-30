import logging

import numpy as np
import onnx
from onnx.onnx_ml_pb2 import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml.lbir.datatype_pb2 import Datatype as LBIRDatatype
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


def node_has_attr(node: NodeProto, attr: str) -> bool:
    for x in node.attribute:
        if x.name == attr:
            return True
    return False


def transform_matmul(model: ModelWrapper, node) -> bool:
    """
    First checks that the MatMul node has two QTensors as inputs and is followed by
    an Add node and an (optional) activation (ReLU) node.
    """
    model_changed = False
    b_err_str = (
        f"Could not transform node: {node.op_type}, with inputs: {node.input},"
        f" and outputs: {node.output}."
    )
    pre = model.find_direct_predecessors(node)
    if len(pre) != 2:
        logging.warning(f"{b_err_str} Because it does not have exactly 2 inputs.")
        return False
    if not (node_has_attr(pre[0], "values") or node_has_attr(pre[1], "values")):
        logging.warning(
            f"{b_err_str} Because neither of the inputs has values (weights)."
        )
        return False
    weights_node, input_node = (
        (pre[0], pre[1]) if node_has_attr(pre[0], "values") else (pre[1], pre[0])
    )
    if input_node.op_type == "QDense":
        input_qtensor = DenseConfig.FromString(
            onnx.helper.get_node_attr_value(input_node, "qdense")
        ).output
        inputs = input_node.output
    else:
        input_qtensor = QTensor.FromString(
            onnx.helper.get_node_attr_value(input_node, "qtensor")
        )
        inputs = input_node.input
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
    densecfg = DenseConfig(
        thresh=thresh,
        kernel=QTensor.FromString(
            onnx.helper.get_node_attr_value(weights_node, "qtensor")
        ),
        input=input_qtensor,
        output=QTensor.FromString(
            onnx.helper.get_node_attr_value(output_node, "qtensor")
        ),
        activation=activation,
    )
    for n in (node, weights_node, bias_node, add_node, output_node):
        model.graph.node.remove(n)
    if input_node.op_type == "QTensor":
        model.graph.node.remove(input_node)
    if activation_node is not None:
        model.graph.node.remove(activation_node)
    kwargs = _denseconfig_to_kwargs(densecfg)
    new_node = onnx.helper.make_node(
        op_type="QDense",
        inputs=inputs,
        outputs=output_node.output,
        domain="chisel4ml",
        qdense=densecfg.SerializeToString(),
        **kwargs,
    )
    model.graph.node.extend([new_node])
    model_changed = True
    return model_changed


class QONNXToLBIR(Transformation):
    """
    Transforms the QONNX nodes to LBIR nodes.
    """

    def apply(self, model: ModelWrapper):
        model_changed = False
        for node in model.graph.node:
            # Check for the "root" computational nodes
            if node.op_type == "MatMul":
                model_changed = transform_matmul(model, node)
                break
            if node.op_type == "Conv":
                raise NotImplementedError
        return model, model_changed


class WeightQuantToQTensor(Transformation):
    """
    Transforms the Quant nodes with weights to LBIR QTensor.
    """

    def apply(self, model: ModelWrapper):
        model_changed = False
        for node in model.graph.node:
            # We search for quant nodes with an initialized input[0] (weights)
            if (
                node.op_type in ("Quant", "BipolarQuant")
                and model.get_initializer(node.input[0]) is not None
            ):
                if node.op_type == "Quant":
                    weight_init, scale_init, zp_init, bw_init = (
                        node.input[0],
                        node.input[1],
                        node.input[2],
                        node.input[3],
                    )
                    weights = model.get_initializer(weight_init)
                    shift = _scale_to_shift(
                        np.atleast_1d(model.get_initializer(scale_init)),
                        weights.shape[1],
                    )
                    offset = (
                        np.atleast_1d(model.get_initializer(zp_init))
                        .astype(int)
                        .tolist()
                    )
                    bitwidth = int(model.get_initializer(bw_init).item())
                    quantization = LBIRDatatype.QuantizationType.UNIFORM
                    signed = node.attribute[2].i == 1
                else:
                    weight_init, scale_init = node.input[0], node.input[1]
                    weights = model.get_initializer(weight_init)
                    shift = _scale_to_shift(
                        np.atleast_1d(model.get_initializer(scale_init)),
                        weights.shape[1],
                    )
                    offset = [0]
                    bitwidth = 1
                    quantization = LBIRDatatype.QuantizationType.BINARY
                    signed = 1

                qt = QTensor(
                    dtype=LBIRDatatype(
                        quantization=quantization,
                        signed=signed,
                        bitwidth=bitwidth,
                        shift=shift,
                        offset=offset,
                    ),
                    shape=weights.T.shape,  # transpose to get "right" shape for lbir
                    values=weights.flatten().tolist(),
                )

                successor = model.find_direct_successors(node)
                assert len(successor) == 1
                # This node is (most likely) here becuase it was inserted when
                # converting from QKeras to qonnx (see QONNX converter). If this is not
                # a case, then this function will incorrectly delete this operation.
                if successor[0].op_type == "Mul":
                    outputs = successor[0].output
                    model.graph.node.remove(successor[0])
                else:
                    outputs = node.output
                model.graph.node.remove(node)
                kwargs = _qtensor_to_kwargs(qt)
                new_node = onnx.helper.make_node(
                    op_type="QTensor",
                    inputs=[],
                    outputs=outputs,
                    domain="chisel4ml",
                    qtensor=qt.SerializeToString(),
                    **kwargs,
                )
                model.graph.node.extend([new_node])
                model_changed = True
                break
        return model, model_changed


class QuantToQTensor(Transformation):
    """
    Transforms the other Quant nodes to LBIR QTensors.
    """

    def apply(self, model: ModelWrapper):
        model_changed = False
        for node in model.graph.node:
            # Quant nodes without predecessors are weight nodes
            if (
                node.op_type in ("Quant", "BipolarQuant")
                and model.get_initializer(node.input[0]) is None
            ):
                if node.op_type == "Quant":
                    scale_init, zp_init, bw_init = (
                        node.input[1],
                        node.input[2],
                        node.input[3],
                    )
                    shift = _scale_to_shift(
                        np.atleast_1d(model.get_initializer(scale_init)),
                        model.get_tensor_shape(node.input[0])[1],
                    )
                    offset = (
                        np.atleast_1d(model.get_initializer(zp_init))
                        .astype(int)
                        .tolist()
                    )
                    bitwidth = int(model.get_initializer(bw_init).item())
                    quantization = LBIRDatatype.QuantizationType.UNIFORM
                    signed = node.attribute[2].i == 1
                    rounding_mode = onnx.helper.get_node_attr_value(
                        node, "rounding_mode"
                    )
                else:
                    scale_init = node.input[1]
                    shift = _scale_to_shift(
                        np.atleast_1d(model.get_initializer(scale_init)),
                        model.get_tensor_shape(node.input[0])[1],
                    )
                    offset = [0]
                    bitwidth = 1
                    quantization = LBIRDatatype.QuantizationType.BINARY
                    signed = 1
                    rounding_mode = "NONE"

                qt = QTensor(
                    dtype=LBIRDatatype(
                        quantization=quantization,
                        signed=signed,
                        bitwidth=bitwidth,
                        shift=shift,
                        offset=offset,
                    ),
                    shape=model.get_tensor_shape(node.input[0])[
                        1:
                    ],  # we remove the batch dimension
                    rounding_mode=rounding_mode,
                )
                model.graph.node.remove(node)
                kwargs = _qtensor_to_kwargs(qt)
                new_node = onnx.helper.make_node(
                    op_type="QTensor",
                    inputs=[node.input[0]],
                    outputs=node.output,
                    domain="chisel4ml",
                    qtensor=qt.SerializeToString(),
                    **kwargs,
                )
                model.graph.node.extend([new_node])
                model_changed = True
                break
        return model, model_changed


class UnquantizedBiasToQTensor(Transformation):
    """
    If Bias is left unquantized it tries to quantize it and replaces
    the initializer node with a QTensor.
    """

    def apply(self, model: ModelWrapper):
        model_changed = False
        for node in model.graph.node:
            if node.op_type == "Add":
                pre = model.find_direct_predecessors(node)
                # Only check adds that follow a MatMul or Conv (Conv/Dense layers)
                if (
                    len(pre) == 1
                    and pre[0].op_type in ("MatMul", "Conv")
                    and len(node.input) == 2
                ):
                    init0 = model.get_initializer(node.input[0])
                    init1 = model.get_initializer(node.input[1])
                    if not ((init0 is not None) or (init1 is not None)):
                        return model, False
                    bias, param_input = (
                        (init0, node.input[0])
                        if init0 is not None
                        else (init1, node.input[1])
                    )
                    qt = _numpy_to_qtensor(bias)
                    kwargs = _qtensor_to_kwargs(qt)
                    new_node = onnx.helper.make_node(
                        op_type="QTensor",
                        inputs=[],
                        outputs=[param_input],
                        domain="chisel4ml",
                        qtensor=qt.SerializeToString(),
                        **kwargs,
                    )
                    model.graph.node.extend([new_node])
                    model.del_initializer(param_input)
                    model_changed = True
                    break
        return model, model_changed


class UnquantizedOutputToQTensor(Transformation):
    """
    If Output is left unquantized it tries to quantize it and replaces
    the initializer node with a QTensor.
    """

    def apply(self, model: ModelWrapper):
        model_outputs = [out.name for out in model.graph.output]
        model_changed = False
        for node in model.graph.node:
            if node.output[0] in model_outputs and node.op_type != "QTensor":
                assert (
                    len(node.output) == 1
                ), "There should be only one output per node."
                pre = model.find_direct_predecessors(node)
                if pre[0].op_type != "QTensor":
                    logging.warning(
                        (
                            f"Output {node.output[0]} left unquantized, adding default "
                            "8-bit quantization."
                        )
                    )
                    qt = QTensor(
                        dtype=LBIRDatatype(
                            quantization=LBIRDatatype.QuantizationType.UNIFORM,
                            signed=True,
                            bitwidth=8,
                            shift=[0] * model.get_tensor_shape(node.output[0])[1],
                            offset=[0],
                        ),
                        shape=model.get_tensor_shape(node.output[0])[
                            1:
                        ],  # we remove the batch dimension
                        rounding_mode="NONE",
                    )
                    kwargs = _qtensor_to_kwargs(qt)
                    ind = model_outputs.index(node.output[0])
                    new_link_name = f"pre_quant_out_{ind}"
                    new_node = onnx.helper.make_node(
                        op_type="QTensor",
                        inputs=[new_link_name],
                        outputs=[node.output[0]],
                        domain="chisel4ml",
                        qtensor=qt.SerializeToString(),
                        **kwargs,
                    )
                    node.output[0] = new_link_name
                    model.graph.node.extend([new_node])
                    model_changed = True
        return model, model_changed


class InputReluQTensorToQTensor(Transformation):
    """
    If input is a Relu followed by a QTensor we replace this
    with just the QTensor (that is unsigned UNIFORM quantized)
    """

    def apply(self, model: ModelWrapper):
        model_inputs = [inp.name for inp in model.graph.input]
        model_changed = False
        for node in model.graph.node:
            if (
                len(node.input) > 0
                and node.input[0] in model_inputs
                and node.op_type == "Relu"
            ):
                suc = model.find_direct_successors(node)
                if suc[0].op_type == "QTensor":
                    qtensor = QTensor.FromString(
                        onnx.helper.get_node_attr_value(suc[0], "qtensor")
                    )
                    assert not qtensor.dtype.signed
                    model.graph.node.remove(node)
                    suc[0].input[0] = node.input[0]
                    model_changed = True
                    break
        return model, model_changed


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
