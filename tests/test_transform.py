from difflib import context_diff
from difflib import ndiff

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml import transform
from chisel4ml.lbir.datatype_pb2 import Datatype
from chisel4ml.lbir.qtensor_pb2 import QTensor


def UQ(b, s=None, o=[0]):
    return Datatype(
        quantization=Datatype.QuantizationType.UNIFORM,
        signed=False,
        bitwidth=b,
        shift=s,
        offset=o,
    )


def SQ(b, s=None, o=[0]):
    return Datatype(
        quantization=Datatype.QuantizationType.UNIFORM,
        signed=True,
        bitwidth=b,
        shift=s,
        offset=o,
    )


def BQ(s=None, o=[0]):
    return Datatype(
        quantization=Datatype.QuantizationType.BINARY,
        signed=True,
        bitwidth=1,
        shift=s,
        offset=o,
    )


def test_bnn_simple_model_transform(bnn_simple_model):
    lbir_model = transform.qkeras_to_lbir(bnn_simple_model)
    lbir_ref = lbir.Model(
        name=lbir_model.name,
        layers=[
            lbir.LayerWrap(
                dense=lbir.DenseConfig(
                    thresh=QTensor(
                        dtype=SQ(16, [0]),
                        shape=[4],
                        values=[-1.0, -2.0, -0.0, -1.0],
                    ),
                    kernel=QTensor(
                        dtype=BQ([0] * 4),
                        shape=[4, 3],
                        values=[1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1],
                    ),
                    input=QTensor(
                        dtype=BQ([0]),
                        shape=[3],
                    ),
                    output=QTensor(
                        dtype=BQ([0]),
                        shape=[4],
                    ),
                    activation=lbir.Activation.BINARY_SIGN,
                )
            ),
            lbir.LayerWrap(
                dense=lbir.DenseConfig(
                    thresh=QTensor(
                        dtype=SQ(16, [0]),
                        shape=[1],
                        values=[-1],
                    ),
                    kernel=QTensor(
                        dtype=BQ([0]),
                        shape=[1, 4],
                        values=[-1, 1, -1, -1],
                    ),
                    input=QTensor(
                        dtype=BQ([0]),
                        shape=[4],
                    ),
                    output=QTensor(
                        dtype=BQ([0]),
                        shape=[1],
                    ),
                    activation=lbir.Activation.BINARY_SIGN,
                )
            ),
        ],
    )
    model_str = repr(lbir_model).splitlines(keepends=True)
    ref_str = repr(lbir_ref).splitlines(keepends=True)
    diffs = list(context_diff(model_str, ref_str))
    diffs_with_text = list(ndiff(model_str, ref_str))
    assert len(diffs) == 0, (
        "Generated LBIR does not match reference. See the diff,"
        " where + signifies additions to reference, - the "
        "removals\n" + "".join(diffs_with_text)
    )
