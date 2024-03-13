import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pytest_cases import fixture

from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer


@fixture(scope="session")
def mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


@fixture(scope="session")
def audio_data():
    train_ds, info = tfds.load(
        "speech_commands",
        split="train",
        with_info=True,
        shuffle_files=False,
        as_supervised=True,
    )
    val_ds = tfds.load(
        "speech_commands", split="validation", shuffle_files=False, as_supervised=True
    )
    test_ds = tfds.load(
        "speech_commands", split="test", shuffle_files=False, as_supervised=True
    )

    label_names = []
    for name in info.features["label"].names:
        print(name, info.features["label"].str2int(name))
        label_names = label_names[:] + [name]

    def get_frames(x):
        npads = (32 * 512) - x.shape[0]
        frames = np.pad(x, (0, npads)).reshape([32, 512])
        frames = np.round(((frames / 2**15)) * 2047 * 0.8)
        return frames.reshape(32, 512)

    def train_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(train_ds),
        )

    def val_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(val_ds),
        )

    def test_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(test_ds),
        )

    train_set = tf.data.Dataset.from_generator(  # noqa: F841
        train_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )

    val_set = tf.data.Dataset.from_generator(  # noqa: F841
        val_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    test_set = tf.data.Dataset.from_generator(  # noqa: F841
        test_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    return [
        train_set,
        val_set,
        test_set,
        label_names,
        len(train_ds),
        len(val_ds),
        len(test_ds),
    ]


@fixture(scope="session")
def audio_data_preproc():
    train_ds, info = tfds.load(
        "speech_commands",
        split="train",
        with_info=True,
        shuffle_files=False,
        as_supervised=True,
    )
    val_ds = tfds.load(
        "speech_commands", split="validation", shuffle_files=False, as_supervised=True
    )
    test_ds = tfds.load(
        "speech_commands", split="test", shuffle_files=False, as_supervised=True
    )

    label_names = []
    for name in info.features["label"].names:
        print(name, info.features["label"].str2int(name))
        label_names = label_names[:] + [name]

    fft_layer = FFTLayer(FFTConfig(fft_size=512, num_frames=32, win_fn=np.hamming(512)))
    lmfe_layer = LMFELayer(LMFEConfig(fft_size=512, num_frames=32, num_mels=20))

    def preproc(sample):
        return lmfe_layer(fft_layer(np.expand_dims(sample, axis=0)))

    def get_frames(x):
        npads = (32 * 512) - x.shape[0]
        frames = np.pad(x, (0, npads)).reshape([32, 512])
        frames = np.round(((frames / 2**15)) * 2047 * 0.8)
        lmfre = preproc(frames.reshape(32, 512))
        return lmfre[0]  # remove the batch dimension (which is 1 in this function)

    def train_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(train_ds),
        )

    def val_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(val_ds),
        )

    def test_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(test_ds),
        )

    train_set = tf.data.Dataset.from_generator(  # noqa: F841
        train_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 20, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )

    val_set = tf.data.Dataset.from_generator(  # noqa: F841
        val_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 20, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    test_set = tf.data.Dataset.from_generator(  # noqa: F841
        test_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 20, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    return [
        train_set,
        val_set,
        test_set,
        label_names,
        len(train_ds),
        len(val_ds),
        len(test_ds),
    ]
