from chisel4ml import optimize, transform

import tensorflow as tf

import subprocess
import os


def hardware(model: tf.keras.Model, gen_dir="./gen/"):
    "Generate verilog code from model"
    opt_model = optimize.qkeras_model(model)
    lbir_model = transform.qkeras2lbir(opt_model)
    pbfile = os.path.join(gen_dir, "model.pb")
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    with open(pbfile, "wb") as f:
        f.write(lbir_model.SerializeToString())

    cmd = ["java", "-jar", "bin/chisel4ml.jar", gen_dir, "model.pb"]
    subprocess.run(cmd, capture_output=True, check=True)
