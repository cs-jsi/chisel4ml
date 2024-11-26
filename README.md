# Chisel4ml
Chisel4ml is an open-source library for generating highly-parallel dataflow style hardware implementations of Deeply Quantized Neural Networks. These types of networks are trained using frameworks such as [Brevitas](https://github.com/Xilinx/brevitas) and [QKeras](https://github.com/google/qkeras). However, any training framework is supported as long as it export [QONNX](https://github.com/fastmachinelearning/qonnx).

## Instalation: from pip
1. pip install chisel4ml.
2. Download a matching jar from github relases.
3. To test first run `java -jar chisel4ml.jar` (You can change the port and temporary directory using -p and -d (use --help for info)
4. Paste the Python code bellow into a file and run `python script.py`

```
import numpy as np
import qkeras
import tensorflow as tf
from chisel4ml import optimize, generate

w1 = np.array([[1, 2, 3, 4], [-4, -3, -2, -1], [2, -1, 1, 1]])
b1 = np.array([1, 2, 0, 1])
w2 = np.array([-1, 4, -3, -1]).reshape(4, 1)
b2 = np.array([2])

x = x_in = tf.keras.layers.Input(shape=3)
x = qkeras.QActivation(
    qkeras.quantized_bits(bits=4, integer=3, keep_negative=True)
)(x)
x = qkeras.QDense(
    4,
    kernel_quantizer=qkeras.quantized_bits(
        bits=4, integer=3, keep_negative=True, alpha=np.array([0.5, 0.25, 1, 0.25])
    ),
)(x)
x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
x = qkeras.QDense(
    1,
    kernel_quantizer=qkeras.quantized_bits(
        bits=4, integer=3, keep_negative=True, alpha=np.array([0.125])
    ),
)(x)
x = qkeras.QActivation(qkeras.quantized_relu(bits=3, integer=3))(x)
model = tf.keras.Model(inputs=[x_in], outputs=[x])
model.compile()
model.layers[2].set_weights([w1, b1])
model.layers[4].set_weights([w2, b2])
data = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [4.0, 4.0, 4.0],
        [7.0, 7.0, 7.0],
        [6.0, 0.0, 7.0],
        [3.0, 3.0, 3.0],
        [7.0, 0.0, 0.0],
        [0.0, 7.0, 0.0],
        [0.0, 0.0, 7.0],
    ]
)


opt_model = optimize.qkeras_model(model)
accelerators, lbir_model = generate.accelerators(
    model,
    minimize="delay"
)
circuit = generate.circuit(opt_model)
for x in data:
    sw_res = opt_model.predict(np.expand_dims(x, axis=0))
    hw_res = circuit(x)
    assert np.array_equal(sw_res.flatten(), hw_res.flatten())
circuit.delete_from_server()
```
This will generate a circuit of a simple two layer fully-connected neural network, and store it in `/tmp/.chisel4ml/circuit0`.
If you have verilator installed you can also add the argument: `use_verilator=True` in the `generate.circuit` function. In the first case only a firrtl file be generated (this can be converted to verilog using firtool), if you use verilator, however, a SystemVerilog file will also be created.

chisel4ml also supports convolutional layers and maxpool layers. It also has some support for calculating FFTs and log-mel feature energy (audio features) in hardware.

## Installation: from source
1. Install [mill build tool](https://mill-build.com/mill/Intro_to_Mill.html).
2. Install [python](https://www.python.org/downloads/) 3.8-3.10
3. Create environment `python -m venv venv/`
4. Activate environment (Linux)`source venv/bin/activate`
    - Windows `.\venv\Scripts\activate`
5. Upgrade pip `python -m pip install --upgrade pip`
6. Install chisel4ml `pip install -ve .[dev]`
7. Build Python protobuf code `make`
8. Build Scala code `mill chisel4ml.assembly`
10. In another terminal run tests `pytest --use-verilator -n auto`
    - The `--use-verilator` flag is optional if you have verilator installed, however it is highly recommended, since it is much faster.


## ScalaDocs
To create ScalaDocs run `mill chisel4ml.docJar` and they will be generated to `out/chisel4ml/docJar.dest/javadoc`.
