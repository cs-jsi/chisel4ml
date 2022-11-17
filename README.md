# Chisel4ml
Chisel4ml is an open-source library for generating dataflow architectures inspired by the hls4ml library.

![Tests on master](https://github.com/cs-jsi/chisel4ml/actions/workflows/tests.yml/badge.svg?branch=master)

Run "sbt assembly" to build a standalone .jar executable.

## Instalation

1. Install [sbt](https://www.scala-sbt.org/download.html).
2. Install [python](https://www.python.org/downloads/) 3.6 or higher
3. Create environment `python -m venv venv/`
4. Activate environment (Linux)`source venv/bin/activate`
    - Windows `.\venv\Scripts\activate`
5. Upgrade pip `python -m pip install --upgrade pip`
6. Install base requirements `pip install -r requirements.txt`
7. Install development requirements `pip install -r requirements_dev.txt`
8. Build Python protobuf code `make`
9. Build Scala code `sbt assembly`
10. Run tests `pytest -svv`
