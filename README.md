# Chisel4ml
Chisel4ml is an open-source library for generating dataflow architectures inspired by the hls4ml library.

![Tests](https://github.com/jurevreca12/chisel4ml/actions/workflows/tests.yml/badge.svg)

Run "sbt assembly" to build a standalone .jar executable.

To update the protobuffer generated descriptions run:
    protoc -I=. --python_out=./lbir_python --mypy_out=./lbir_python lbir.proto and
    scalapbc --scala_out=flat_package:./scala/ lbir.proto

in the root/chisel4ml directory.
