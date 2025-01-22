# Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from onnx import helper
from onnx import TensorProto
# from onnxruntime_extensions import onnx_op
# from onnxruntime_extensions import PyCustomOpDef
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp


# @onnx_op(
#     op_type="chisel4ml.preprocess::FFTreal",
#     attrs={
#         "fft_size": PyCustomOpDef.dt_int64,
#         "num_frames": PyCustomOpDef.dt_int64,
#         "win_fn": PyCustomOpDef.dt_float,
#     },
# )
def FFTreal_onnx_op(x, **kwargs):
    win_fn = kwargs["win_fn"]
    fft_size = kwargs["fft_size"]
    num_frames = kwargs["num_frames"]
    results = []
    for frame in x:
        res = np.fft.fft(frame * win_fn, norm="backward", axis=-1).real
        results.append(np.expand_dims(res, axis=-1).astype(np.float32))
    ret = np.reshape(
        np.array(results),
        [len(x), num_frames, fft_size, 1],
    )
    return ret


class FFTreal(CustomOp):
    def get_nodeattr_types(self):
        return {
            "fft_size": (128,),
            "num_frames": (8,),
            "win_fn": (np.array([]),),
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        return helper.make_node(
            "Cast",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            to=int(TensorProto.FLOAT),
        )

    def get_integer_datatype(self, model):
        return DataType["FLOAT32"]

    def get_scaled_integer_datatype(self, model):
        return DataType["FLOAT32"]

    def get_output_dtype(self, model):
        return DataType["FLOAT32"]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], DataType["FLOAT32"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_tensor = context[node.input[0]]
        win_fn = self.get_nodeattr("win_fn")
        results = []
        for x in inp_tensor:
            res = np.fft.fft(x * win_fn, norm="backward", axis=-1).real
            results.append(np.expand_dims(res, axis=-1).astype(np.float32))
        ret = np.reshape(
            np.array(results),
            [len(inp_tensor), self.cfg.num_frames, self.cfg.fft_size, 1],
        )
        context[node.output[0]] = ret

    def verify_node(self):
        pass
