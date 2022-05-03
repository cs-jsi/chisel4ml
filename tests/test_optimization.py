from chisel4ml.optimizations.qkeras_remove_dead_layers import QKerasRemoveDeadLayersOptimization
from chisel4ml.optimizations.qkeras_activation_fold import QKerasActivationFold
import tensorflow as tf
import qkeras
import pytest


@pytest.mark.parametrize("layer", [
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.InputLayer()
    ])
def test_remove_dead_layer_opt(layer):
    """
        Tests the optimization removes all the inactive layers as it should.
    """
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


def test_activation_fold_opt():
    """
        The activation fold in the seperate activation layer into the active layer.
    """
    l0 = qkeras.QDense(64, kernel_quantizer=qkeras.binary())
    l1 = qkeras.QActivation(qkeras.binary())
    opt = QKerasActivationFold()
    opt_layers = opt([l0, l1])
    assert (len(opt_layers) == 1 and 
            type(opt_layers[0]) is qkeras.QDense and
            type(opt_layers[0].activation) is type(l1.activation))
