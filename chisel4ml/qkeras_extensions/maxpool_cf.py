import tensorflow as tf


class MaxPool2dCF(tf.keras.layers.Layer):
    """
    Performs MaxPooling2D as if it were the classic layer
    tf.keras.layers.MaxPooling2D(data_format="channels_first").
    The reason for defininig this is because tf doesn't support
    this op on CPU.

    TODO: Remove this after restructiring to qonnx
    """

    def __init__(
        self, pool_size=(2, 2), input_format="channels_first", activation=None
    ):
        super(MaxPool2dCF, self).__init__()
        if input_format == "channels_last" or input_format == "channels_first":
            self.input_format = input_format
        else:
            raise ValueError(f"Invalid input_format: {input_format}")
        self.data_format = self.input_format
        self.pool_size = pool_size
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size)
        self.activation = activation

    def build(self, input_shape):
        assert len(input_shape) == 4

    def call(self, inputs):
        # inputs: NHWC -> NCHW -> flatten = output
        #         0123 -> 0312
        #         NCHW -> NHWC
        #         0123 -> 0231
        x = inputs
        if self.input_format == "channels_first":
            x = tf.transpose(inputs, perm=(0, 2, 3, 1))
        x = self.maxpool(x)
        if self.input_format == "channels_first":
            x = tf.transpose(x, perm=(0, 3, 1, 2))
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "pool_size": self.pool_size,
            "input_format": self.input_format,
            "activation": self.activation,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(
            pool_size=config["pool_size"],
            input_format=config["input_format"],
            activation=config["activation"],
        )
