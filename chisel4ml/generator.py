from chisel4ml.optimizer import optimize_model
from chisel4ml.transformer import transform_model_to_lbir

import tensorflow as tf

import subprocess


def generate_verilog(model: tf.keras.Model):
    "Generate verilog code from model"
    opt_model = optimize_model(model)
    lbir_model = transform_model_to_lbir(opt_model)
    with open("test.pb", encoding='utf-8') as f:
        f.write(lbir_model.SerializeToString())

    subprocess.run(["java", "-jar", "bin/chisel4ml.jar", "test.pb"])
