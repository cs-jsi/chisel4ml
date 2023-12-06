import tensorflow as tf


class FlattenChannelwise(tf.keras.layers.Layer):
    """
    Flattens the output one channel at a time (as opposed to regular keras flatten
    that interleaves the outputs)
    """

    def __init__(self, data_format="channels_last"):
        super(FlattenChannelwise, self).__init__()
        assert data_format == "channels_last"
        self.data_format = data_format

    def build(self, input_shape):
        assert len(input_shape) == 4
        self.out_shape = input_shape[1] * input_shape[2] * input_shape[3]

    def call(self, inputs):
        # inputs: NHWC -> NCHW -> flatten = output
        #         0123 -> 0312
        output = tf.transpose(inputs, [0, 3, 1, 2])
        return tf.reshape(output, (tf.shape(output)[0], self.out_shape))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "data_format": self.data_format,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(
            data_format=config["data_format"],
        )
