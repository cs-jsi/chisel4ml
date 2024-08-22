import numpy as np
import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from chisel4ml.lbir.datatype_pb2 import Datatype as LBIRDatatype
from chisel4ml.lbir.qtensor_pb2 import QTensor


class QONNXToLBIR(Transformation):
    """
    Transforms the QONNX nodes to LBIR nodes.
    """

    def apply(self, model: ModelWrapper):
        raise NotImplementedError
        return model, False


class QuantToQTensor(Transformation):
    """
    Transforms the Quant nodes to LBIR QTensor.
    """

    def apply(self, model: ModelWrapper):
        model_changed = False
        for node in model.graph.node:
            # Quant nodes without predecessors are weight nodes
            if node.op_type == "Quant" and model.find_direct_predecessors(node) is None:
                weight_init, scale_init, zp_init, bw_init = (
                    node.input[0],
                    node.input[1],
                    node.input[2],
                    node.input[3],
                )
                weights = model.get_initializer(weight_init)
                shift = _scale_to_shift(
                    np.atleast_1d(model.get_initializer(scale_init))
                )
                offset = np.atleast_1d(model.get_initializer(zp_init)).tolist()
                bitwidth = model.get_initializer(bw_init).item()
                qt = QTensor(
                    dtype=LBIRDatatype(
                        quantization=LBIRDatatype.QuantizationType.UNIFORM,
                        signed=node.attribute[2].i == 1,
                        bitwidth=bitwidth,
                        shift=shift,
                        offset=offset,
                    ),
                    shape=weights.shape,
                    values=weights.flatten().tolist(),
                )
                new_node = onnx.helper.make_node(
                    op_type="QTensor",
                    inputs=[],
                    outputs=node.output,
                    domain="chisel4ml",
                    qtensor=qt.SerializeToString(),
                )
                model.graph.node.extend([new_node])
                model.graph.node.remove(node)
                model_changed = True
                break
        return model, model_changed


def _scale_to_shift(scale):
    return (1 / scale).astype(int).tolist()
