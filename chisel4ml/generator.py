from chisel4ml.optimizer import optimize_model
from chisel4ml.transformer import transform_model_to_lbir

import tensorflow as tf

import subprocess
from subprocess import STDOUT, PIPE


def generate_verilog(model: tf.keras.Model, pbfile):
    "Generate verilog code from model"
    opt_model = optimize_model(model)
    lbir_model = transform_model_to_lbir(opt_model)
    with open(pbfile, "wb") as f:
        f.write(lbir_model.SerializeToString())

    cmd = ["java", "-jar", "bin/chisel4ml.jar", pbfile]
    subprocess.run(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
