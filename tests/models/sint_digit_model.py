import qkeras
import tensorflow as tf
from pytest_cases import case
from sklearn import datasets
from sklearn.model_selection import train_test_split

from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


@case(tags="trainable")
def case_sint_digit_model():
    digits_ds = datasets.load_digits()
    data = digits_ds.images.reshape((len(digits_ds.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits_ds.target, test_size=0.2, shuffle=False
    )
    x = x_in = tf.keras.layers.Input(shape=(8, 8, 1))
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=8, keep_negative=False)
    )(x)
    x = QDepthwiseConv2DPermuted(
        kernel_size=[3, 3],
        depth_multiplier=1,
        depthwise_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha=1.0
        ),
    )(x)
    x = qkeras.QActivation(qkeras.quantized_relu(bits=4, integer=4))(x)
    x = tf.keras.layers.Flatten()(x)
    x = qkeras.QDense(
        10,
        kernel_quantizer=qkeras.quantized_bits(
            bits=4, integer=3, keep_negative=True, alpha="auto_po2"
        ),
        use_bias=False,
    )(x)
    x = qkeras.QActivation(
        qkeras.quantized_bits(bits=8, integer=7, keep_negative=True)
    )(x)
    model = tf.keras.Model(inputs=[x_in], outputs=[x])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    NUM_TEST_SAMPLES = 10
    data = {
        "X_train": X_train.reshape(-1, 8, 8, 1),
        "y_train": y_train,
        "X_test": X_test.reshape(-1, 8, 8, 1)[:NUM_TEST_SAMPLES],
        "y_test": y_test[:NUM_TEST_SAMPLES],
    }
    training_info = {"epochs": 10, "batch_size": 128, "callbacks": None}
    return (model, data, training_info)
