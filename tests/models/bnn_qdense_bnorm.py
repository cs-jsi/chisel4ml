import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case


@case(tags="trainable")
def case_bnn_qdense_bnorm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=2))
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.add(qkeras.QDense(3, kernel_quantizer=qkeras.binary(alpha=1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.binary(alpha=1)))
    model.compile(optimizer="adam", loss="squared_hinge", metrics=["accuracy"])
    X_train = np.array(
        [[-1.0, -1.0], [-1.0, +1.0], [+1.0, -1.0], [+1.0, +1.0]]
    )  # noqa: F841
    y_train = np.array([0.0, 1.0, 1.0, 0.0])  # noqa: F841
    data = {"X_train": X_train, "y_train": y_train, "X_test": X_train}
    training_info = {"epochs": 5, "batch_size": 4, "callbacks": None}
    return (model, data, training_info)
