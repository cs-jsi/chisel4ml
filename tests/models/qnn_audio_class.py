import numpy as np
import qkeras
import tensorflow as tf
from pytest_cases import case
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


@case(tags="trainable-gen")
def case_qnn_audio_class(audio_data):
    train_set = audio_data[0]  # noqa: F841
    val_set = audio_data[1]  # noqa: F841
    test_set = audio_data[2]
    label_names = audio_data[3]
    TRAIN_SET_LENGTH = audio_data[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data[5]  # noqa: F841

    EPOCHS = 3  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    input_shape = (32, 512)
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
            qkeras.quantized_bits(12, 11, keep_negative=True, alpha=1),
            input_shape=input_shape,
        )
    )
    model.add(FFTLayer(FFTConfig(fft_size=512, num_frames=32, win_fn=np.hamming(512))))
    model.add(LMFELayer(LMFEConfig(fft_size=512, num_frames=32, num_mels=20)))
    model.add(
        qkeras.QActivation(
            activation=qkeras.quantized_bits(8, 7, keep_negative=True, alpha=1),
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
    callbacks = [pruning_callbacks.UpdatePruningStep()]
    NUM_TEST_ELEMENTS = 3
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
