import warnings

from onnx import helper
from qonnx.transformation.base import Transformation


class ExtractQuantizedBiasFromConv(Transformation):
    """
    Extracts the Bias that is quantized from a Conv(Transpose) node and inserts it
    behind the Conv(Transpose) node as an Add node.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type in ["Conv", "ConvTranspose"]:
                # Check if the node has a bias input
                if len(n.input) > 2:
                    # Extract bias
                    bias = model.find_producer(n.input[2])
                    if bias is None or bias.op_type != "Quant":
                        warnings.warn(f"Could not extract bias from node {n}")
                        continue

                    # Insert bias as Add node behind the Conv node
                    out_shape = model.get_tensor_shape(n.output[0])
                    # Reshape bias tensor
                    add_shape = [1] * len(out_shape)
                    # ToDo: this must change to "add_shape[-1] = bias.shape[0]" when
                    #  the channels last layout comes around.
                    bias_shape = model.get_tensor_shape(n.input[2])
                    if len(bias_shape) < 2:
                        bias_shape = [
                            out_shape[0],
                            bias_shape[0],
                        ]  # add batch dimension
                        model.set_tensor_shape(n.input[2], bias_shape)
                    add_shape[1] = bias_shape[1]
                    conv_new_output_name = f"conv_{node_ind}_out"
                    add_node = helper.make_node(
                        op_type="Add",
                        inputs=[conv_new_output_name, n.input[2]],
                        outputs=[n.output[0]],
                    )
                    graph.node.insert(node_ind, add_node)

                    # Repoint Conv output and remove bias tensor
                    n.output[0] = conv_new_output_name
                    n.input.remove(n.input[2])

                    return model, True

        return model, False
