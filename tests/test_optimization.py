import chisel4ml.optimizations as opt
import tensorflow as tf
import pytest


@pytest.mark.parametrize("layer", [
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.InputLayer()
    ])
def test_remove_dead_layer_opt(layer):
    assert opt.qkeras_opt_factory(layer)([layer]) == [], \
            f"The optimization {opt.qkeras_opt_factory(layer)} was suppose to optimize away the {layer} layer. The " \
            f"problem is either with the optimization itself, or the qkeras_opt_factory code in optimizations/" \
            f"__init__.py."


def test_check_num_layers_decorator():
    """
        The optimization for the input layer expects a list of length one. This test checks whether the
        _check_num_layers decorator in the optimizations/qkeras_optimization.py works correctly.
    """
    layer = tf.keras.layers.InputLayer()
    with pytest.raises(AssertionError):
        opt.qkeras_opt_factory(layer)([layer, layer])
