# This file tests the behavior of qkeras quantizers, to make sure they are inline with the expected
# behavior.
import numpy as np
import qkeras
import tensorflow as tf


def test_quantized_bits_unsigned_int():
    """ We define a quantized bits quantizer to act like an unsigned integer quantizer. """
    uquant = qkeras.quantized_bits(bits=4, 
                                   integer=4, 
                                   keep_negative=False)
    assert tf.reduce_all(tf.math.equal(np.arange(0, 16), uquant(np.arange(0, 16))))
    # np.full creates 10 copies of 15: (15, 15, 15 ... )
    assert tf.reduce_all(tf.math.equal(np.full((10), 15), uquant(np.arange(16, 26))))
    assert tf.reduce_all(tf.math.equal(np.full((32), 0), uquant(np.arange(-32,0))))


def test_quantized_bits_signed_int():
    """ We define a quantized bits quantizer to act like an unsigned integer quantizer. """
    uquant = qkeras.quantized_bits(bits=4, 
                                   integer=3, 
                                   keep_negative=True)
    assert tf.reduce_all(tf.math.equal(np.arange(0, 8), uquant(np.arange(0, 8))))
    assert tf.reduce_all(tf.math.equal(np.full((10), 7), uquant(np.arange(8, 18))))
    assert tf.reduce_all(tf.math.equal(np.arange(-8, 0), uquant(np.arange(-8, 0))))
    for i in [0.1, 0.2, 0.3, 0.4]:
        assert tf.reduce_all(tf.math.equal(np.arange(0, 8), uquant(np.arange(0, 8)+i)))
    for i in [0.6, 0.7, 0.8, 0.9]:
        assert tf.reduce_all(tf.math.equal(np.clip(np.arange(0, 8)+1, 0, 7), uquant(np.arange(0, 8)+i)))
    for i in [0.1, 0.2, 0.3, 0.4]:
        assert tf.reduce_all(tf.math.equal(np.arange(-8, 0), uquant(np.arange(-8, 0)+i)))
    for i in [0.6, 0.7, 0.8, 0.9]:
        assert tf.reduce_all(tf.math.equal(np.clip(np.arange(-8, 0)+1, -8, 0), uquant(np.arange(-8, 0)+i)))
