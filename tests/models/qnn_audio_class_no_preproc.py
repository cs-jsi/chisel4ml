import qkeras
import tensorflow as tf
from pytest_cases import case
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


@case(tags="trainable-gen")
def case_qnn_audio_class_no_preproc(audio_data_preproc):
    train_set = audio_data_preproc[0]  # noqa: F841
    val_set = audio_data_preproc[1]  # noqa: F841
    test_set = audio_data_preproc[2]  # noqa: F841
    label_names = audio_data_preproc[3]
    TRAIN_SET_LENGTH = audio_data_preproc[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data_preproc[5]  # noqa: F841
    EPOCHS = 3  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    input_shape = (32, 20, 1)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    pruning_params = {
        "pruning_schedule": pruning_schedule.ConstantSparsity(
            0.90, begin_step=2000, frequency=100
        )
    }

    model = tf.keras.models.Sequential()
    model.add(
        qkeras.QActivation(
            activation=qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1),
            input_shape=input_shape,
        ),
    )
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=1,
            use_bias=True,
            bias_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha=1
            ),
            depthwise_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=5, integer=5)))
    model.add(
        QDepthwiseConv2DPermuted(
            kernel_size=[3, 3],
            depth_multiplier=2,
            use_bias=True,
            bias_quantizer=qkeras.quantized_bits(
                bits=8, integer=7, keep_negative=True, alpha=1
            ),
            depthwise_quantizer=qkeras.quantized_bits(
                bits=4, integer=3, keep_negative=True, alpha="auto_po2"
            ),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(FlattenChannelwise())
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                8,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                bias_quantizer=qkeras.quantized_bits(
                    bits=8, integer=7, keep_negative=True, alpha=1
                ),
            ),
            **pruning_params,
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3)))
    model.add(
        prune.prune_low_magnitude(
            qkeras.QDense(
                num_labels,
                kernel_quantizer=qkeras.quantized_bits(
                    bits=4, integer=3, keep_negative=True, alpha="auto_po2"
                ),
                use_bias=True,
                bias_quantizer=qkeras.quantized_bits(
                    bits=8, integer=7, keep_negative=True, alpha=1
                ),
            ),
            **pruning_params,
        )
    )
    model.add(
        qkeras.QActivation(qkeras.quantized_bits(bits=8, integer=7, keep_negative=True))
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.build()
    callbacks = [pruning_callbacks.UpdatePruningStep()]
    NUM_TEST_ELEMENTS = 10
    data = {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set.take(NUM_TEST_ELEMENTS),
    }
    training_info = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "callbacks": callbacks,
        "train_len": TRAIN_SET_LENGTH,
        "val_len": VAL_SET_LENGTH,
    }
    return (model, data, training_info)
