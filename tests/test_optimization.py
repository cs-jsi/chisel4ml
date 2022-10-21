from chisel4ml.optimizations.qkeras_remove_dead_layers import QKerasRemoveDeadLayersOptimization
from chisel4ml.optimizations.qkeras_activation_fold import QKerasActivationFold
from chisel4ml.optimizations.qkeras_bn_qdense_binary_fuse import QKerasBNQDenseBinaryFuse
from chisel4ml import optimize
from math import isclose

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import qkeras
import pytest


def test_check_num_layers_functionality():
    """
        The optimization for the input layer expects a list of length one. This test makes sure whether the
        the optimizations check for this.
    """
    layer = tf.keras.layers.InputLayer()
    opt = QKerasRemoveDeadLayersOptimization()
    with pytest.raises(AssertionError):
        opt([layer, layer])


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


def test_sint_mnist_qdense_relu_opt(sint_mnist_qdense_relu):
    """ Tests if the model performs (approximatly) as well after optimization, as before optimization. """
    (_, _), (x_test, y_test) = mnist.load_data()
    image_vector_size = 28*28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype('float32')
    y_test = tf.one_hot(y_test, 10)
    (_, acc) = sint_mnist_qdense_relu.evaluate(x_test, y_test, verbose=0)
    opt_model = optimize.qkeras_model(sint_mnist_qdense_relu)
    (_, acc_opt) = opt_model.evaluate(x_test, y_test, verbose=0)
    assert isclose(acc, acc_opt, abs_tol=0.03), \
        f"The prediction of the optimized model should be with in 3 percent of the original model. Numerical " \
        f"instability can account for such small differences, bigger differences are likely some other failure. " \
        f"The original sint_mnist_qdense_relu model had accuracy of {acc} and the optimized {acc_opt}."


def test_sint_mnist_qdense_relu_pruned_opt(sint_mnist_qdense_relu_pruned):
    """ Tests if the pruned model performs (approximatly) as well after optimization, as before optimization. """
    (_, _), (x_test, y_test) = mnist.load_data()
    image_vector_size = 28*28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype('float32')
    y_test = tf.one_hot(y_test, 10)
    (_, acc) = sint_mnist_qdense_relu_pruned.evaluate(x_test, y_test, verbose=0)
    opt_model = optimize.qkeras_model(sint_mnist_qdense_relu_pruned)
    (_, acc_opt) = opt_model.evaluate(x_test, y_test, verbose=0)
    assert isclose(acc, acc_opt, abs_tol=0.05), \
        f"The prediction of the optimized model should be with in 5 percent of the original model. Numerical " \
        f"instability can account for such small differences, bigger differences are likely some other failure. " \
        f"The original sint_mnist_qdense_relu_pruned model had accuracy of {acc} and the optimized {acc_opt}."
