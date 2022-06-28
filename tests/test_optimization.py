from chisel4ml.optimizations.qkeras_remove_dead_layers import QKerasRemoveDeadLayersOptimization
from chisel4ml.optimizations.qkeras_activation_fold import QKerasActivationFold
from chisel4ml.optimizations.qkeras_bn_qdense_binary_fuse import QKerasBNQDenseBinaryFuse
from chisel4ml import optimize

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
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
    assert hasattr(opt([layer])[0], 'c4ml_remove_layer'), \
        f"The optimization {opt} was suppose to optimize away the {layer} layer."


def test_check_num_layers_functionality():
    """
        The optimization for the input layer expects a list of length one. This test makes sure whether the
        the optimizations check for this.
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
    assert (hasattr(opt_layers[1], 'c4ml_remove_layer') and
            type(opt_layers[0]) is qkeras.QDense and
            isinstance(opt_layers[0].activation, type(l1.activation)))


def test_bn_qdense_binary_fuse_opt(bnn_qdense_bn_sign_act):
    """ Tests the fusing of binary qdense layer with a batch normalization layer. """
    org_model = bnn_qdense_bn_sign_act
    new_model = tf.keras.models.clone_model(bnn_qdense_bn_sign_act)
    new_model.build((None, 2))
    new_model.compile(optimizer='adam', loss='squared_hinge')
    new_model.set_weights(org_model.get_weights())
    l0 = new_model.layers[0]
    l1 = new_model.layers[1]
    l2 = new_model.layers[2]
    opt = QKerasBNQDenseBinaryFuse()
    assert opt.is_applicable([l0, l1, l2])
    opt_layers = opt([l0, l1, l2])
    assert hasattr(opt_layers[1], 'c4ml_remove_layer')
    for i0 in [+1., -1.]:
        for i1 in [+1., -1.]:
            org_res = org_model.predict(np.array([i0, i1]).reshape(1, 2))
            opt_res = opt_layers[2](opt_layers[0](np.array([i0, i1]).reshape(1, 2)))
            assert tf.reduce_all(tf.math.equal(org_res, opt_res)), \
                "There seems to be a problem with the BatchNorm fuse operation for binary kerenels. The original " \
                f"model predicted {org_res} but the optimized version predicted {opt_res} for inputs [{i0},{i1}]." \
                f"The original parameters are qdense layer:{org_model.layers[0].weights}, {org_model.layers[0].bias}" \
                f" and for batchnorm: {org_model.layers[1].get_weights()} and the optimized weights are: " \
                f"{opt_layers[0].weights}, {opt_layers[0].bias}."


def test_bnn_mnist_model_opt(bnn_mnist_model):
    """ The optimization of this model should yield a model that produces the same results. """
    opt_model = optimize.qkeras_model(bnn_mnist_model)
    (_, _), (x_test, y_test) = mnist.load_data()
    image_vector_size = 28*28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype('float32')
    for i in range(0, 10):
        org_res = bnn_mnist_model.predict(x_test[i].reshape(1, 784))
        opt_res = opt_model.predict(x_test[i].reshape(1, 784))
        assert tf.reduce_all(tf.math.equal(org_res, opt_res)), \
            f"The original model predicted the result: {org_res}, where as the optimized model predicted: {opt_res}." \
            f"The results differed on mnist test image index: {i}."
