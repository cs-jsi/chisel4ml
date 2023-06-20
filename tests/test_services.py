import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from chisel4ml import generate
from chisel4ml import optimize


def test_compile_service(bnn_simple_model):
    """Test if the compile service is working correctly."""
    opt_model = optimize.qkeras_model(bnn_simple_model)
    circuit = generate.circuit(opt_model, is_simple=False)
    assert circuit is not None


def test_run_service(bnn_simple_model):
    """Tests if the run service (simulation) is working correctly)."""
    opt_model = optimize.qkeras_model(bnn_simple_model)
    circuit = generate.circuit(opt_model, is_simple=False)
    assert circuit is not None
    for i in [-1.0, 1.0]:
        for j in [-1.0, 1.0]:
            for k in [-1.0, 1.0]:
                sw_res = opt_model.predict(np.array([[i, j, k]]))
                hw_res = circuit(np.array([i, j, k]))
                assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), (
                    f"The software model predicted the result {sw_res}, where as the"
                    f" hardware model predicted {hw_res}. Something is wrong here. The"
                    f" stated results are for the inputs {i}, {j}, {k}. "
                )


def test_run_service_2(bnn_simple_bweight_model):
    """Tests if the run service (simulation) is working correctly for binary weight
    layers.
    """
    opt_model = optimize.qkeras_model(bnn_simple_bweight_model)
    circuit = generate.circuit(opt_model, is_simple=False, gen_vcd=True)
    assert circuit is not None
    for inp in [
        [36.0, 22.0, 3.0],
        [6.0, 18.0, 5.0],
        [6.0, 22.0, 3.0],
        [255.0, 127.0, 255.0],
        [0.0, 0.0, 0.0],
        [255.0, 255.0, 255.0],
    ]:
        sw_res = opt_model.predict(np.array([inp]))
        hw_res = circuit(np.array(inp))
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), (
            f"The software model predicted the result {sw_res}, where as the hardware"
            f" model predicted {hw_res}. Something is wrong here. The stated results"
            f" are for the inputs: {inp}. "
        )


@pytest.mark.skip(reason="to expensive to run")
def test_run_service_3(bnn_mnist_model):
    """Tests if the run service (simulation) is working correctly on more complicated
    models that have BinaryWeightDense layers.
    """
    (_, _), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28 * 28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype("float32")
    y_test = tf.one_hot(y_test, 10)
    y_test = np.where(y_test < 0.1, -1.0, 1.0)

    opt_model = optimize.qkeras_model(bnn_mnist_model)
    circuit = generate.circuit(
        opt_model, is_simple=True, use_verilator=True, gen_vcd=True
    )
    assert circuit is not None
    for i in range(0, 10):
        sw_res = opt_model.predict(x_test[i].reshape(1, 784))
        hw_res = circuit(x_test[i])
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), (
            f"The software model predicted the result {sw_res}, where as the hardware"
            f" model predicted {hw_res}. Something is wrong here. The stated results"
            f" are for the mnist image index {i}. "
        )


def test_run_service_4(sint_simple_noscale_model):
    """Tests if non-binary quantized network is implemented correctly in hardware
    (by simulation).
    """
    x_test = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0],
            [6.0, 0.0, 12.0],
            [3.0, 3.0, 3.0],
            [15.0, 0.0, 0.0],
            [0.0, 15.0, 0.0],
            [0.0, 0.0, 15.0],
        ]
    )

    opt_model = optimize.qkeras_model(sint_simple_noscale_model)
    circuit = generate.circuit(opt_model, is_simple=False)
    assert circuit is not None
    for i in range(0, 10):
        sw_res = opt_model.predict(x_test[i].reshape(1, 3))
        hw_res = circuit(x_test[i])
        assert tf.reduce_all(tf.math.equal(sw_res, hw_res)), (
            f"The software model predicted the result {sw_res}, where as the hardware"
            f" model predicted {hw_res}. Something is wrong here. The stated results"
            f" are for the inputs {x_test[i]}. "
        )


def test_run_service_5(sint_simple_model):
    """Tests if quantized network with scale factors is implemented correctly in
    hardware (by simulation).
    """
    x_test = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [4.0, 4.0, 4.0],
            [7.0, 7.0, 7.0],
            [6.0, 0.0, 7.0],
            [3.0, 3.0, 3.0],
            [7.0, 0.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 7.0],
        ]
    )

    opt_model = optimize.qkeras_model(sint_simple_model)
    circuit = generate.circuit(opt_model, is_simple=False)
    assert circuit is not None
    for i in range(0, 10):
        sw_res = opt_model.predict(x_test[i].reshape(1, 3))
        hw_res = circuit(x_test[i])
        assert tf.reduce_all(np.isclose(sw_res, hw_res, atol=1.0)), (
            f"The software model predicted the result {sw_res}, where as the hardware"
            f" model predicted {hw_res}. Something is wrong here. The stated results"
            f" are for the inputs {x_test[i]}. "
        )


@pytest.mark.skip(reason="to expensive to run")
def test_run_service_6(sint_mnist_qdense_relu):
    """Test a more complex non-binary model."""
    (_, _), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28 * 28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype("float32")
    y_test = tf.one_hot(y_test, 10)

    opt_model = optimize.qkeras_model(sint_mnist_qdense_relu)
    circuit = generate.circuit(opt_model, is_simple=False)
    assert circuit is not None
    for i in range(0, 10):
        sw_res = opt_model.predict(x_test[i].reshape(1, 784))
        if isinstance(sw_res, tuple):
            sw_res = sw_res[0]
        sw_index = np.where(sw_res == np.amax(sw_res))[1][0]
        hw_res = circuit(x_test[i])
        hw_index = np.where(hw_res == np.amax(hw_res))[0][0]
        assert sw_index == hw_index, (
            f"The software model predicted the result {sw_res}, where as the hardware"
            f" model predicted {hw_res}. The index of the bigest element should be the"
            " same but instead the bigest element of the software model is at index:"
            f" {sw_index}, and the hardware model at index: {hw_index}. Something is"
            f" wrong here. The stated results are for the mnist test image index {i}. "
        )


@pytest.mark.skip(reason="to expensive to run")
def test_run_service_7(sint_mnist_qdense_relu_pruned):
    """Tests if a pruned non-binary model works correctly. Note that the optimizations
    change the model somewhat, so this test is not really through.
    """
    (_, _), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    image_vector_size = 28 * 28
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype("float32")
    y_test = tf.one_hot(y_test, 10)
    y_test = np.where(y_test < 0.1, -1.0, 1.0)

    opt_model = optimize.qkeras_model(sint_mnist_qdense_relu_pruned)
    circuit = generate.circuit(opt_model, is_simple=False)
    assert circuit is not None
    for i in range(0, 6):
        sw_res = opt_model.predict(x_test[i].reshape(1, 784))
        if isinstance(sw_res, tuple):
            sw_res = sw_res[0]
        sw_index = np.where(sw_res == np.amax(sw_res))[1][0]
        hw_res = circuit(x_test[i])
        hw_index = np.where(hw_res == np.amax(hw_res))[0][0]
        assert sw_index == hw_index, (
            f"The software model predicted the result {sw_res}, where as the hardware"
            f" model predicted {hw_res}. The index of the bigest element should be the"
            " same but instead the bigest element of the software model is at index:"
            f" {sw_index}, and the hardware model at index: {hw_index}. Something is"
            f" wrong here. The stated results are for the mnist test image index {i}. "
        )
