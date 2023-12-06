import numpy as np
import qkeras
import tensorflow as tf


class QDepthwiseConv2DPermuted(tf.keras.layers.Layer):
    """
    Wraps a QDepthwiseConv2D object with permutation of output channels. This allows
    more efficient computation with current Conv PU. The default QDepthwiseConv2D
    groups the outputs by channel, this version instead groups them by filters.
    """

    def __init__(
        self,
        kernel_size,
        depth_multiplier=1,
        use_bias=True,
        depthwise_quantizer=None,
        bias_quantizer=None,
        data_format="channels_last",
    ):
        super(QDepthwiseConv2DPermuted, self).__init__()
        self.dwconv = qkeras.QDepthwiseConv2D(
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            use_bias=use_bias,
            depthwise_quantizer=depthwise_quantizer,
            bias_quantizer=bias_quantizer,
            data_format=data_format,
        )
        self.kernel_size = kernel_size
        self.depth = depth_multiplier
        self.use_bias = use_bias
        self.depthwise_quantizer = depthwise_quantizer
        self.bias_quantizer = bias_quantizer
        self.data_format = data_format
        self.ch_axis = 1 if self.data_format == "channels_first" else -1

    def build(self, input_shape):
        self.dwconv.build(input_shape)
        inp_ch = input_shape[self.ch_axis]
        self.channel_order = (
            np.arange(inp_ch * self.depth)
            .reshape(inp_ch, self.depth)
            .transpose()
            .flatten()
        )
        self.depthwise_quantizer_internal = self.dwconv.depthwise_quantizer_internal
        self.activation = self.dwconv.activation
        self.bias_quantizer_internal = self.dwconv.bias_quantizer_internal
        self.bias = self.dwconv.bias

    @property
    def depthwise_kernel(self):
        return self.dwconv.depthwise_kernel

    def call(self, inputs):
        dwout = self.dwconv(inputs)
        return tf.gather(dwout, self.channel_order, axis=self.ch_axis)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "kernel_size": self.kernel_size,
            "depth_multiplier": self.depth,
            "use_bias": self.use_bias,
            "depthwise_quantizer": self.depthwise_quantizer,
            "bias_quantizer": self.bias_quantizer,
            "data_format": self.data_format,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(
            kernel_size=config["kernel_size"],
            depth_multiplier=config["depth_multiplier"],
            use_bias=config["use_bias"],
            depthwise_quantizer=config["depthwise_quantizer"],
            bias_quantizer=config["bias_quantizer"],
            data_format=config["data_format"],
        )
