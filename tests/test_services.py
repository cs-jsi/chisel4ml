from chisel4ml import elaborate
import numpy as np
import tensorflow as tf
import pytest

from tensorflow.keras.datasets import mnist


def test_compile_service(bnn_simple_model):
    """ Test if the compile service is working correctly. """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle is not None


def test_run_service(bnn_simple_model):
    """ Tests if the run service (simulation) is working correctly). """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle is not None
    for i in [-1.0, 1.0]:
        for j in [-1.0, 1.0]:
            for k in [-1.0, 1.0]:
                sw_res = bnn_simple_model.predict(np.array([[i, j, k]]))
                hw_res = epp_handle(np.array([i, j, k]))
                assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), \
                    f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
                    f"{hw_res}. Something is wrong here. The stated results are for the inputs {i}, {j}, {k}. "


def test_run_service_2(bnn_simple_bweight_model):
    """ Tests if the run service (simulation) is working correctly for binary weight layers. """
    epp_handle = elaborate.qkeras_model(bnn_simple_bweight_model)
    assert epp_handle is not None
    for inp in [[36, 22, 3], [6, 18, 5], [6, 22, 3], [255, 127, 255], [0, 0, 0], [255, 255, 255]]:
        sw_res = bnn_simple_bweight_model.predict(np.array([inp]))
        hw_res = epp_handle(np.array(inp))
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. Something is wrong here. The stated results are for the inputs: {inp}. "


def test_run_service_3(bnn_mnist_model):
    """
        Tests if the run service (simulation) is working correctly on more complicated models that have
        BinaryWeightDense layers.
    """
    (_, _), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28*28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype('float32')
    y_test = tf.one_hot(y_test, 10)
    y_test = np.where(y_test < 0.1, -1., 1.)

    epp_handle = elaborate.qkeras_model(bnn_mnist_model)
    assert epp_handle is not None
    for i in range(0, 10):
        sw_res = bnn_mnist_model.predict(x_test[i].reshape(1, 784))
        hw_res = epp_handle(x_test[i])
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. Something is wrong here. The stated results are for the mnist image index {i}. "


def test_run_service_4(sint_simple_noscale_model):
    """ Tests if non-binary quantized network is implemented correctly in hardware (by simulation). """
    x_test = np.array([[0, 0, 0],
                       [0, 1, 2],
                       [2, 1, 0],
                       [4, 4, 4],
                       [15, 15, 15],
                       [6, 0, 12],
                       [3, 3, 3],
                       [15, 0, 0],
                       [0, 15, 0],
                       [0, 0, 15]])

    epp_handle = elaborate.qkeras_model(sint_simple_noscale_model)
    assert epp_handle is not None
    for i in range(0, 10):
        sw_res = sint_simple_noscale_model.predict(x_test[i].reshape(1, 3))
        hw_res = epp_handle(x_test[i])
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. Something is wrong here. The stated results are for the inputs {x_test[i]}. "


def test_run_service_5(sint_simple_model):
    """ Tests if quantized network with scale factors is implemented correctly in hardware (by simulation). """
    x_test = np.array([[0, 0, 0],
                       [0, 1, 2],
                       [2, 1, 0],
                       [4, 4, 4],
                       [15, 15, 15],
                       [6, 0, 12],
                       [3, 3, 3],
                       [15, 0, 0],
                       [0, 15, 0],
                       [0, 0, 15]])

    epp_handle = elaborate.qkeras_model(sint_simple_model)
    assert epp_handle is not None
    for i in range(0, 10):
        sw_res = sint_simple_model.predict(x_test[i].reshape(1, 3))
        hw_res = epp_handle(x_test[i])
        assert tf.reduce_all(np.isclose(sw_res, hw_res, atol=1.0)), \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. Something is wrong here. The stated results are for the inputs {x_test[i]}. "


def test_run_service_6(sint_mnist_qdense_relu):
    """ Test a more complex non-binary model. """
    (_, _), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28*28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype('float32')
    y_test = tf.one_hot(y_test, 10)

    epp_handle = elaborate.qkeras_model(sint_mnist_qdense_relu)
    assert epp_handle is not None
    for i in range(0, 10):
        sw_res = sint_mnist_qdense_relu.predict(x_test[i].reshape(1, 784))
        sw_index = np.where(sw_res == np.amax(sw_res))[1][0]
        hw_res = epp_handle(x_test[i])
        hw_index = np.where(hw_res == np.amax(hw_res))[0][0]
        assert sw_index == hw_index, \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. The index of the bigest element should be the same but instead the bigest element of " \
            f"the software model is at index: {sw_index}, and the hardware model at index: {hw_index}. " \
            f"Something is wrong here. The stated results are for the mnist test image index {i}. "


def test_run_service_7(sint_mnist_qdense_relu_pruned):
    """ 
        Tests if a pruned non-binary model works correctly. Note that the optimizations change the model 
        somewhat, so this test is not really through. 
    """
    (_, _), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28*28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype('float32')
    y_test = tf.one_hot(y_test, 10)
    y_test = np.where(y_test < 0.1, -1., 1.)

    epp_handle = elaborate.qkeras_model(sint_mnist_qdense_relu_pruned)
    assert epp_handle is not None
    for i in range(0, 6):
        sw_res = sint_mnist_qdense_relu_pruned.predict(x_test[i].reshape(1, 784))
        sw_index = np.where(sw_res == np.amax(sw_res))[1][0]
        hw_res = epp_handle(x_test[i])
        hw_index = np.where(hw_res == np.amax(hw_res))[0][0]
        assert sw_index == hw_index, \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. The index of the bigest element should be the same but instead the bigest element of " \
            f"the software model is at index: {sw_index}, and the hardware model at index: {hw_index}. " \
            f"Something is wrong here. The stated results are for the mnist test image index {i}. "
