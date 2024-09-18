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
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp


class lmfe(CustomOp):
    def get_nodeattr_types(self):
        return {
            "fft_size": (int, True, 128),
            "num_frames": (int, True, 8),
            "num_mels": (int, True, 20),
            "filter_banks": (np.array([]),),
        }

    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        fft_size = model.get_initializer(self.onnx_node.input[2]).item()
        num_mels = model.get_initializer(self.onnx_node.input[3]).item()
        temp_tensor_val = np.ones((fft_size, num_mels))
        temp_tensor_name = model.make_new_valueinfo_name()

        model.set_initializer(temp_tensor_name, temp_tensor_val)
        return helper.make_node(
            "MatMul",
            inputs=[self.onnx_node.input[0], temp_tensor_name],
            outputs=[self.onnx_node.output[0]],
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
        filter_banks = self.get_nodeattr("filter_banks")
        fft_size = self.get_nodeattr("fft_size")
        half = (fft_size // 2) + 1
        fft_res = inp_tensor[:, :, 0:half]
        mag_frames = (fft_res + 1) ** 2  # 1 is added for numerical stability
        mels = np.matmul(mag_frames, filter_banks.T)
        log_mels = np.log2(mels, dtype=np.float32)
        ret = np.floor(log_mels)
        context[node.output[0]] = ret

    def verify_node(self):
        pass
