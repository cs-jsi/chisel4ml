import pytest
import qkeras
import tensorflow as tf
from pytest_cases import case
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


@case(tags="trainable")
@pytest.mark.skip(reason="to expensive to run")
def case_sint_mnist_qdense_relu_pruned(mnist_data):
    """An MNIST model with only fully-connected layers (no conv) that is pruned with TF
    model optimization toolkit.
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

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.90, begin_step=2000, frequency=100
        )
    }

    kernel_quant_params = {
        "bits": 4,
        "integer": 3,
        "keep_negative": True,
        "alpha": "auto_po2",
    }

    model = tf.keras.models.Sequential()
    # We don't loose any info here since mnist are 8-bit gray-scale images. We just add
    # this quantization to explicitly encode this for the chisel4ml optimizer.
    model.add(tf.keras.layers.Input(shape=image_vector_size))
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=8, integer=8)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                32,
                kernel_quantizer=qkeras.quantized_bits(**kernel_quant_params),
                use_bias=True,
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                32,
                kernel_quantizer=qkeras.quantized_bits(**kernel_quant_params),
                use_bias=True,
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                32,
                kernel_quantizer=qkeras.quantized_bits(**kernel_quant_params),
                use_bias=True,
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))

    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                num_classes,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                activation="softmax",
            ),
            **pruning_params,
        )
    )
    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    callbacks = [pruning_callbacks.UpdatePruningStep()]
    data = {"X_train": x_train, "y_train": y_train, "X_test": x_test, "y_test": y_test}
    training_info = {"epochs": 10, "batch_size": 64, "callbacks": callbacks}
    return (model, data, training_info)
