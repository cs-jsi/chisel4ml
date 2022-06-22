from chisel4ml import optimize, transform
import chisel4ml.lbir.lbir_pb2 as lbir


def test_qkeras_transform(bnn_mnist_model):
    opt_model = optimize.qkeras_model(bnn_mnist_model)
    lbir_model = transform.qkeras2lbir(opt_model)
    layers = lbir_model.layers

    UNIFORM = lbir.Datatype.QuantizationType.UNIFORM
    BINARY = lbir.Datatype.QuantizationType.BINARY
    BINARY_SIGN = lbir.Activation.Function.BINARY_SIGN

    # We expect a Binarized Neural Network with 4 dense layer. 1st uses a binary-weights,
    # and the rest are fully binarized.

    # ltype
    for lay in layers:
        assert lay.ltype is lbir.Layer.DENSE, "There should be only DENSE layers left here."

    # use_bias
    for lay in layers:
        assert lay.use_bias is True

    # biases
    assert layers[0].biases.dtype.quantization is UNIFORM
    assert int(layers[0].biases.dtype.bitwidth) == 32
    assert int(layers[0].biases.dtype.scale) == 1
    assert int(layers[0].biases.dtype.offset) == 0
    assert len(layers[0].biases.shape) > 0
    assert len(layers[0].biases.values) > 0
    for lay in layers[1:-1]:
        assert lay.biases.dtype.quantization is UNIFORM
        assert int(lay.biases.dtype.bitwidth) == 32
        assert int(lay.biases.dtype.scale) == 1
        assert int(lay.biases.dtype.offset) == 0
        assert len(lay.biases.shape) > 0
        assert len(lay.biases.values) > 0

    assert layers[-1].biases.dtype.quantization is UNIFORM
    assert int(layers[-1].biases.dtype.bitwidth) == 32
    assert int(layers[-1].biases.dtype.scale) == 1
    assert int(layers[-1].biases.dtype.offset) == 0
    assert len(layers[-1].biases.shape) > 0
    assert len(layers[-1].biases.values) > 0

    # weights
    for lay in layers:
        assert lay.weights.dtype.quantization is BINARY
        assert int(lay.weights.dtype.bitwidth) == 1
        assert int(lay.weights.dtype.scale) == 1
        assert int(lay.weights.dtype.offset) == 0
        assert len(lay.weights.shape) > 0
        assert len(lay.weights.values) > 0
        for val in lay.weights.values:
            assert (int(val) == 1) or (int(val) == -1)

    # input
    assert layers[0].input.dtype.quantization is UNIFORM, "The first layers input is uniformly quantized."
    for lay in layers[1:]:
        assert lay.input.dtype.quantization is BINARY, \
                "Except for the first layer these should all have binary input quantization."
        assert int(lay.input.dtype.bitwidth) == 1
        assert int(lay.input.dtype.scale) == 1
        assert int(lay.input.dtype.offset) == 0
        assert len(lay.input.shape) > 0
        assert len(lay.input.values) == 0  # Input values should be empty.

    # activation
    for lay in layers:
        assert lay.activation.fn is BINARY_SIGN
        assert int(lay.activation.bitwidth) == 1

    # out_shape
    for lay in layers:
        assert len(lay.out_shape) > 0
