import logging

import numpy as np
import onnx
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from chisel4ml.lbir.datatype_pb2 import Datatype as LBIRDatatype
from chisel4ml.lbir.qtensor_pb2 import QTensor
from chisel4ml.transforms.qonnx_utils import _numpy_to_qtensor
from chisel4ml.transforms.qonnx_utils import _qtensor_to_kwargs
from chisel4ml.transforms.qonnx_utils import _scale_to_shift
from chisel4ml.transforms.qonnx_utils import get_lbir_shape
from chisel4ml.transforms.qonnx_utils import replace_tensor
from chisel4ml.transforms.transform_conv import transform_conv
from chisel4ml.transforms.transform_fftreal import transform_fftreal
from chisel4ml.transforms.transform_lmfe import transform_lmfe
from chisel4ml.transforms.transform_matmul import transform_matmul
from chisel4ml.transforms.transform_maxpool import transform_maxpool


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
            elif node.op_type == "Conv":
                model_changed = transform_conv(model, node)
            elif node.op_type == "MaxPool":
                model_changed = transform_maxpool(model, node)
            elif node.op_type == "FFTreal":
                model_changed = transform_fftreal(model, node)
            elif node.op_type == "lmfe":
                model_changed = transform_lmfe(model, node)
            if model_changed:
                return model, True
        return model, False


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
                    scale = model.get_initializer(scale_init)
                    shift = _scale_to_shift(np.atleast_1d(scale))

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
                    scale = model.get_initializer(scale_init)
                    shift = _scale_to_shift(np.atleast_1d(scale))
                    offset = [0]
                    bitwidth = 1
                    quantization = LBIRDatatype.QuantizationType.BINARY
                    signed = 1
                successor = model.find_direct_successors(node)
                assert len(successor) == 1
                old_layout = model.get_tensor_layout(node.output[0])
                new_shape = get_lbir_shape(
                    old_shape=weights.shape, old_layout=old_layout, is_weight=True
                )
                exec_context = model.make_empty_exec_context()
                oxe.execute_node(node, exec_context, model.graph)
                adjusted_weights = exec_context[node.output[0]] / scale
                qt = QTensor(
                    dtype=LBIRDatatype(
                        quantization=quantization,
                        signed=signed,
                        bitwidth=bitwidth,
                        shift=shift,
                        offset=offset,
                    ),
                    shape=new_shape,
                    values=adjusted_weights.flatten().tolist(),
                )
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
                        np.atleast_1d(model.get_initializer(scale_init))
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
                        np.atleast_1d(model.get_initializer(scale_init))
                    )
                    offset = [0]
                    bitwidth = 1
                    quantization = LBIRDatatype.QuantizationType.BINARY
                    signed = 1
                    rounding_mode = "NONE"

                qtshape = model.get_tensor_shape(node.input[0])
                if len(qtshape) > 1:
                    # we remove the batch dimension if its present
                    qtshape = qtshape[1:]
                qt = QTensor(
                    dtype=LBIRDatatype(
                        quantization=quantization,
                        signed=signed,
                        bitwidth=bitwidth,
                        shift=shift,
                        offset=offset,
                    ),
                    shape=qtshape,
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
                    # Bias should always be signed (we invert it to get thresholds)
                    qt.dtype.signed = True
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
                if pre is None or pre[0].op_type != "QTensor":
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
                            shift=[0] * len(model.get_tensor_shape(node.output[0])),
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


class AddInputOrOutputQTensorToReshape(Transformation):
    """
    Reshape nodes should have either an input or output QTensor.
    We add the other QTensor to make life easier for oter transformations."""

    def apply(self, model: ModelWrapper):
        for ind, node in enumerate(model.graph.node):
            if node.op_type == "Reshape":
                pre = model.find_direct_predecessors(node)
                suc = model.find_direct_successors(node)
                assert len(pre) == 1
                assert len(suc) == 1
                if pre[0].op_type == "QTensor" and suc[0].op_type == "QTensor":
                    continue
                qtensor_node = pre[0] if pre[0].op_type == "QTensor" else suc[0]
                qt = QTensor.FromString(
                    onnx.helper.get_node_attr_value(qtensor_node, "qtensor")
                )
                new_link_name = f"reshape_quant_{ind}"
                tensor_tmp = (
                    node.output[0] if pre[0].op_type == "QTensor" else node.input[0]
                )
                tensor_vi = model.get_tensor_valueinfo(tensor_tmp)
                new_val_info = onnx.helper.make_value_info(
                    name=new_link_name, type_proto=tensor_vi.type
                )
                qtshape = model.get_tensor_shape(tensor_tmp)
                if len(qtshape) > 1:
                    # we remove the batch dimension if its present
                    qtshape = qtshape[1:]
                new_qt = QTensor(
                    dtype=qt.dtype,
                    shape=qtshape,
                    values=qt.values,  # should be empty
                    rounding_mode=qt.rounding_mode,
                    layout=qt.layout,
                )
                model.graph.value_info.append(new_val_info)
                if pre[0].op_type == "QTensor":
                    inputs = [node.output[0]]
                    outputs = [new_link_name]
                    suc[0].input[0] = new_link_name
                elif suc[0].op_type == "QTensor":
                    inputs = [pre[0].output[0]]
                    outputs = [new_link_name]
                    node.input[0] = new_link_name

                kwargs = _qtensor_to_kwargs(new_qt)
                new_node = onnx.helper.make_node(
                    op_type="QTensor",
                    inputs=inputs,
                    outputs=outputs,
                    domain="chisel4ml",
                    qtensor=new_qt.SerializeToString(),
                    **kwargs,
                )
                model.graph.node.extend([new_node])
        return model, False


class AddDummyBiasToConv(Transformation):
    """
    If Conv layer has no bias, it adds a zero bias, so that further transformations
    can extract it.
    """

    def apply(self, model):
        model_changed = False
        for ind, node in enumerate(model.graph.node):
            if node.op_type in ["Conv"]:
                # Check if the node has a bias input
                # third (optional) input is a bias
                if len(node.input) < 3:
                    assert len(node.input) == 2
                    weights_shape = model.get_tensor_shape(node.input[1])
                    bias_shape = [weights_shape[0]]
                    zero_bias = np.zeros(bias_shape)
                    model.set_initializer(f"conv_{ind}_bias", zero_bias)
                    node.input.append(f"conv_{ind}_bias")
                    model_changed = True
                    break
        return model, model_changed


class AddFFTrealOutputShape(Transformation):
    def apply(self, model):
        for node in model.graph.node:
            if node.op_type == "FFTreal":
                inp_shape = model.get_tensor_shape(node.input[0])
                model.set_tensor_shape(
                    tensor_name=node.output[0], tensor_shape=inp_shape
                )
        return model, False


class RemoveFlattenNode(Transformation):
    def apply(self, model):
        log_err_str = "Flatten removal transformation failed because: "
        for node in model.graph.node:
            if node.op_type == "Flatten" or node.op_type == "Reshape":
                if node.op_type == "Reshape":
                    shape = model.get_initializer(node.input[1])
                    if not np.array_equal(shape, np.array([-1])):
                        # reshape with shape [-1] equals flatten
                        break
                pre = model.find_direct_predecessors(node)
                suc = model.find_direct_successors(node)
                if len(pre) != 1:
                    logging.warning(f"{log_err_str} More then one input - {len(pre)}")
                    continue
                if len(suc) != 1:
                    logging.warning(f"{log_err_str} More then one output - {len(suc)}")
                    continue
                if pre[0].op_type != "QTensor":
                    logging.warning(f"{log_err_str} Input node not QTensor.")
                    continue
                model.graph.node.remove(node)
                suc[0].input.remove(node.output[0])
                suc[0].input.extend([node.input[0]])

        return model, False


class CleanupQTensors(Transformation):
    "Cleans up QTensors from the graph when the qonnx->lbir transformations are done."

    def replace_input(self, n, orig, new):
        n.input.remove(orig)
        n.input.extend([new])

    def replace_output(self, n, orig, new):
        n.output.remove(orig)
        n.output.extend([new])

    def apply(self, model):
        log_err_str = "QTensor cleanup failed because: "
        for node in model.graph.node:
            if node.op_type != "QTensor":
                continue
            if len(node.input) != 1:
                logging.warning(
                    f"{log_err_str} More then one input - {len(node.input)}"
                )
                continue
            if len(node.output) != 1:
                logging.warning(
                    f"{log_err_str} More then one output - {len(node.output)}"
                )
                continue
            pre = model.find_direct_predecessors(node)
            suc = model.find_direct_successors(node)
            if pre is not None and len(pre) != 1:
                logging.warning(f"{log_err_str} More then one predecessor - {len(pre)}")
                continue
            if suc is not None and len(suc) != 1:
                logging.warning(f"{log_err_str} More then one successor - {len(suc)}")
                continue
            model.graph.node.remove(node)
            if pre is not None and suc is not None:
                self.replace_output(pre[0], node.input[0], node.output[0])
            elif pre is not None:
                self.replace_output(pre[0], node.input[0], node.output[0])
            elif suc is not None:
                self.replace_input(suc[0], node.output[0], node.input[0])
            return model, True
        return model, False


class AutoPadToPad(Transformation):
    "Transforms padding specified with autopad to more explicit pads."

    def autopad_to_pads(self, model, node, autopad):
        if autopad == b"VALID":
            return [0, 0, 0, 0]
        elif autopad == b"SAME_UPPER" or autopad == b"SAME_LOWER":
            ishape = model.get_tensor_shape(node.input[0])  # NCHW
            kshape = model.get_tensor_shape(node.input[1])  # NCHW
            stride = onnx.helper.get_node_attr_value(node, "strides")
            out_width = ishape[-1] // stride[1]
            out_height = ishape[-2] // stride[0]
            # This equation is derived from outshape calc eq
            # see LayerMapping.scala
            wpt = (out_width - 1) * stride[1] - ishape[-1] + kshape[-1]
            hpt = (out_height - 1) * stride[0] - ishape[-2] + kshape[-2]
            hm = hpt // 2
            wm = wpt // 2
            he = hpt % 2
            we = wpt % 2
            if autopad == b"SAME_UPPER":
                return [hm, wm, (hm + he), (wm + we)]
            else:
                return [(hm + he), (wm + we), hm, wm]
        else:
            raise ValueError(f"Value {autopad} not supported.")

    def apply(self, model):
        for node in model.graph.node:
            if node.op_type == "Conv" or node.op_type == "MaxPool":
                has_no_pads = len([x for x in node.attribute if x.name == "pads"]) == 0
                if has_no_pads:
                    autopad = onnx.helper.get_node_attr_value(node, "auto_pad")
                    pads = self.autopad_to_pads(model, node, autopad)
                    attr = onnx.helper.make_attribute("pads", pads)
                    node.attribute.append(attr)
        return model, False


class MergePad(Transformation):
    def transform_pads(self, pads):
        if len(pads) == 4:
            return pads
        if len(pads) == 8:
            # This must be NCHW -> return should be only for HW (spatial axes)
            # https://onnx.ai/onnx/operators/onnx__Pad.html
            # https://onnx.ai/onnx/operators/onnx__Conv.html
            return [pads[2], pads[3], pads[6], pads[7]]
        else:
            raise ValueError

    def apply(self, model):
        log_err_str = "Failed to merge pad into successor node!"
        for node in model.graph.node:
            if node.op_type == "Pad":
                assert len(node.output) == 1
                out_node = model.find_consumer(node.output[0])
                if out_node.op_type in ("Conv", "MaxPool"):
                    mode = onnx.helper.get_node_attr_value(node, "mode")
                    if mode != b"constant":
                        logging.warning(f"{log_err_str} Mode not constant, but {mode}.")
                        continue
                    has_const_val = (
                        len([x for x in node.attribute if x.name == "constant_value"])
                        > 0
                    )
                    if has_const_val:
                        const_val = onnx.helper.get_node_attr_value(
                            node, "constant_value"
                        )
                        if const_val != 0:
                            # Currently only supporting padding with zeros
                            logging.warning(
                                f"{log_err_str} Constant value not 0, but {const_val}"
                            )
                            continue
                    pads = model.get_initializer(node.input[1])
                    transformed_pads = self.transform_pads(pads)
                    pads_attr = onnx.helper.make_attribute("pads", transformed_pads)
                    org_attr = [x for x in out_node.attribute if x.name == "pads"]
                    out_node.attribute.remove(org_attr[0])
                    out_node.attribute.append(pads_attr)
                    replace_tensor(out_node.input, node.output[0], node.input[0])
                    model.graph.node.remove(node)
        return model, False
