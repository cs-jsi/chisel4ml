from chisel4ml import transform
import chisel4ml.lbir.lbir_pb2 as lbir
import pytest

def test_sint_simple_model_transform(sint_simple_model):
    lbir_model = transform.qkeras_to_lbir(sint_simple_model)
    lbir_ref = lbir.Model(
        name = lbir_model.name,
        layers = [
            lbir.Layer(
                ltype = lbir.Layer.Type.DENSE,
                biases = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = True,
                        bitwidth = 32,
                        shift = [0],
                        offset = [0]
                    ),
                    shape = [4],
                    values = [1, 2, 0, 1]
                ),
                weights = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = True,
                        bitwidth = 4,
                        shift = [-1, -2, 0, -2],
                        offset = [0]
                    ),
                    shape = [3, 4],
                    values = [ 1,  2,  3,  4, 
                              -4, -3, -2, -1,
                               2, -1,  1,  1]
                ),
                input = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = True,
                        bitwidth = 4,
                        shift = [0],
                        offset = [0]
                    ),
                    shape = [3]
                ),
                output = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = False,
                        bitwidth = 3,
                        shift = [0],
                        offset = [0]
                    ),
                    shape = [4]
                ),
                activation = lbir.Layer.Activation.RELU
            ),
            lbir.Layer(
                ltype = lbir.Layer.Type.DENSE,
                biases = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = True,
                        bitwidth = 32,
                        shift = [0],
                        offset = [0]
                    ),
                    shape = [1],
                    values = [2]
                ),
                weights = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = True,
                        bitwidth = 4,
                        shift = [-3],
                        offset = [0]
                    ),
                    shape = [4, 1],
                    values = [-1, 4, -3, -1]
                ),
                input = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = False,
                        bitwidth = 3,
                        shift = [0],
                        offset = [0]
                    ),
                    shape = [4]
                ),
                output = lbir.QTensor(
                    dtype = lbir.Datatype(
                        quantization = lbir.Datatype.QuantizationType.UNIFORM,
                        signed = True,
                        bitwidth = 32,
                        shift = [0],
                        offset = [0]
                    ),
                    shape = [1]
                ),
                activation = lbir.Layer.Activation.NO_ACTIVATION
            )
        ]
    )
    assert lbir_model == lbir_ref
