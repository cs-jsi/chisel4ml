from chisel4ml import elaborate
import chisel4ml.lbir.services_pb2 as services
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def test_compile_service(bnn_simple_model):
    """ Test if the compile service is working correctly. """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS


def test_run_service(bnn_simple_model):
    """ Tests if the run service (simulation) is working correctly). """
    epp_handle = elaborate.qkeras_model(bnn_simple_model)
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
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
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
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
    assert epp_handle.reply.err == services.ErrorMsg.ErrorId.SUCCESS
    for i in range(0, 10):
        sw_res = bnn_mnist_model.predict(x_test[i].reshape(1, 784))
        hw_res = epp_handle(x_test[i])
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), \
            f"The software model predicted the result {sw_res}, where as the hardware model predicted " \
            f"{hw_res}. Something is wrong here. The stated results are for the mnist image index {i}. "
