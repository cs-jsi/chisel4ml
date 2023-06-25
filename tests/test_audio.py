import numpy as np
import tensorflow as tf

from chisel4ml import generate


def test_preproc_sine_wave():
    tone_freq = 200
    num_frames = 32
    frame_length = 512

    time_axis = np.linspace(0, 1, 32 * 512)
    sine_wave = np.sin(2 * np.pi * tone_freq * time_axis)
    frames = sine_wave.reshape([num_frames, frame_length])
    frames = np.round((frames + 1) * 2047 * 0.8)

    audio_preproc = generate.circuit(
        opt_model=tf.keras.Model(), get_mfcc=True, use_verilator=True, gen_vcd=True
    )
    ret = audio_preproc(frames)
    assert ret.shape == (32, 20)
