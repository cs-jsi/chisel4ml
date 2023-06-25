from difflib import context_diff
from difflib import ndiff

import chisel4ml.lbir.lbir_pb2 as lbir
from chisel4ml import transform


def UQ(b, s=None, o=[0]):
    return lbir.Datatype(
        quantization=lbir.Datatype.QuantizationType.UNIFORM,
        signed=False,
        bitwidth=b,
        shift=s,
        offset=o,
    )


def SQ(b, s=None, o=[0]):
    return lbir.Datatype(
        quantization=lbir.Datatype.QuantizationType.UNIFORM,
        signed=True,
        bitwidth=b,
        shift=s,
        offset=o,
    )


def BQ(s=None, o=[0]):
    return lbir.Datatype(
        quantization=lbir.Datatype.QuantizationType.BINARY,
        signed=True,
        bitwidth=1,
        shift=s,
        offset=o,
    )


def test_sint_simple_conv_model_transform(sint_simple_conv_model):
    lbir_model = transform.qkeras_to_lbir(sint_simple_conv_model)
    lbir_ref = lbir.Model(
        name=lbir_model.name,
        layers=[
            lbir.Layer(
                ltype=lbir.Layer.Type.CONV2D,
                thresh=lbir.QTensor(
                    dtype=SQ(16, [0] * 2),
                    shape=[2],
                    values=[-1, -2],
                ),
                weights=lbir.QTensor(
                    dtype=SQ(4, [1, 0]),
                    shape=[2, 1, 2, 2],
                    values=[1, 2, 3, 4, -4, -3, -2, -1],
                ),
                input=lbir.QTensor(
                    dtype=SQ(4, [0] * 9),
                    shape=[1, 3, 3],
                ),
                output=lbir.QTensor(
                    dtype=UQ(3, [0] * 8),
                    shape=[2, 2, 2],
                ),
                activation=lbir.Layer.Activation.RELU,
            ),
            lbir.Layer(
                ltype=lbir.Layer.Type.DENSE,
                thresh=lbir.QTensor(
                    dtype=SQ(16, [0]),
                    shape=[1],
                    values=[-2],
                ),
                weights=lbir.QTensor(
                    dtype=SQ(4, [-1]),
                    shape=[1, 8],
                    values=[-1, 4, -3, -1, 2, 3, -3, -2],
                ),
                input=lbir.QTensor(
                    dtype=UQ(3, [0] * 8),
                    shape=[8],
                ),
                output=lbir.QTensor(
                    dtype=SQ(8, [0]),
                    shape=[1],
                ),
                activation=lbir.Layer.Activation.NO_ACTIVATION,
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


def test_sint_simple_model_transform(sint_simple_model):
    lbir_model = transform.qkeras_to_lbir(sint_simple_model)
    lbir_ref = lbir.Model(
        name=lbir_model.name,
        layers=[
            lbir.Layer(
                ltype=lbir.Layer.Type.DENSE,
                thresh=lbir.QTensor(
                    dtype=SQ(16, [0] * 4),
                    shape=[4],
                    values=[-1.0, -2.0, -0.0, -1.0],
                ),
                weights=lbir.QTensor(
                    dtype=SQ(4, [-1, -2, 0, -2]),
                    shape=[4, 3],
                    values=[1, 2, 3, 4, -4, -3, -2, -1, 2, -1, 1, 1],
                ),
                input=lbir.QTensor(
                    dtype=SQ(4, [0] * 3),
                    shape=[3],
                ),
                output=lbir.QTensor(
                    dtype=UQ(3, [0] * 4),
                    shape=[4],
                ),
                activation=lbir.Layer.Activation.RELU,
            ),
            lbir.Layer(
                ltype=lbir.Layer.Type.DENSE,
                thresh=lbir.QTensor(
                    dtype=SQ(16, [0]),
                    shape=[1],
                    values=[-2],
                ),
                weights=lbir.QTensor(
                    dtype=SQ(4, [-3]),
                    shape=[1, 4],
                    values=[-1, 4, -3, -1],
                ),
                input=lbir.QTensor(
                    dtype=UQ(3, [0] * 4),
                    shape=[4],
                ),
                output=lbir.QTensor(
                    dtype=SQ(8, [0]),
                    shape=[1],
                ),
                activation=lbir.Layer.Activation.NO_ACTIVATION,
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


def test_bnn_simple_model_transform(bnn_simple_model):
    lbir_model = transform.qkeras_to_lbir(bnn_simple_model)
    lbir_ref = lbir.Model(
        name=lbir_model.name,
        layers=[
            lbir.Layer(
                ltype=lbir.Layer.Type.DENSE,
                thresh=lbir.QTensor(
                    dtype=SQ(16, [0] * 4),
                    shape=[4],
                    values=[-1.0, -2.0, -0.0, -1.0],
                ),
                weights=lbir.QTensor(
                    dtype=BQ([0] * 4),
                    shape=[4, 3],
                    values=[1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1],
                ),
                input=lbir.QTensor(
                    dtype=BQ([0] * 3),
                    shape=[3],
                ),
                output=lbir.QTensor(
                    dtype=BQ([0] * 4),
                    shape=[4],
                ),
                activation=lbir.Layer.Activation.BINARY_SIGN,
            ),
            lbir.Layer(
                ltype=lbir.Layer.Type.DENSE,
                thresh=lbir.QTensor(
                    dtype=SQ(16, [0]),
                    shape=[1],
                    values=[-1],
                ),
                weights=lbir.QTensor(
                    dtype=BQ([0]),
                    shape=[1, 4],
                    values=[-1, 1, -1, -1],
                ),
                input=lbir.QTensor(
                    dtype=BQ([0] * 4),
                    shape=[4],
                ),
                output=lbir.QTensor(
                    dtype=BQ([0]),
                    shape=[1],
                ),
                activation=lbir.Layer.Activation.BINARY_SIGN,
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
