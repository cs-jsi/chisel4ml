from chisel4ml.optimizations.qkeras_remove_dead_layers import QKerasRemoveDeadLayersOptimization
import tensorflow as tf
import pytest


@pytest.mark.parametrize("layer", [
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.InputLayer()
    ])
def test_remove_dead_layer_opt(layer):
    opt = QKerasRemoveDeadLayersOptimization()
    assert opt([layer]) == [], f"The optimization {opt} was suppose to optimize away the {layer} layer."


def test_check_num_layers_decorator():
    """
        The optimization for the input layer expects a list of length one. This test checks whether the
        _check_num_layers decorator in the optimizations/qkeras_optimization.py works correctly.
    """
    layer = tf.keras.layers.InputLayer()
    opt = QKerasRemoveDeadLayersOptimization()
    with pytest.raises(AssertionError):
        opt([layer, layer])
