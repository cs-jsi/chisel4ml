from chisel4ml import generate, optimize, transform

import tensorflow as tf

import subprocess
from subprocess import STDOUT, PIPE


def hardware(model: tf.keras.Model, pbfile):
    "Generate verilog code from model"
    opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras2lbir(opt_model)
    with open(pbfile, "wb") as f:
        f.write(lbir_model.SerializeToString())

    cmd = ["java", "-jar", "bin/chisel4ml.jar", pbfile]
    subprocess.run(cmd, capture_output=True, check=True)
