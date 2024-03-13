import pytest
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="trainable")
@pytest.mark.skip(reason="to expensive to run")
def case_sint_mnist_qdense_relu(mnist_data):
    """
    Builds a fully-dense (no conv layers) for mnist. The first layer uses unsigned 8
    bit integers as inputs, but the kernels are all quantized to a 4-bit signed
    integer. The activation functions are all ReLU, except for the output activation
    function, which is a softmax (softmax is ignored in hardware). The model achieves
    around 97% accuracy on the MNIST test dataset.
    """
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = mnist_data

    # Flatten the images
    image_vector_size = 28 * 28
    num_classes = 10  # ten unique digits
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_train = x_train.astype("float32")
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    x_test = x_test.astype("float32")

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    model = tf.keras.models.Sequential()
    # We don't loose any info here since mnist are 8-bit gray-scale images. We just add
    # this quantization to explicitly encode this for the chisel4ml optimizer.
    model.add(tf.keras.layers.Input(shape=image_vector_size))
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=8, integer=8)))

    model.add(
        qkeras.QDense(
            32,
            kernel_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        qkeras.QDense(
            32,
            kernel_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        qkeras.QDense(
            32,
            kernel_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        qkeras.QDense(
            num_classes,
            kernel_quantizer=qkeras.quantized_bits(
                bits=3, integer=3, keep_negative=True, alpha="alpha_po2"
            ),
            activation="softmax",
        )
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    data = {"X_train": x_train, "y_train": y_train, "X_test": x_test, "y_test": y_test}
    training_info = {"epochs": 10, "batch_size": 32, "callbacks": None}
    return (model, data, training_info)
