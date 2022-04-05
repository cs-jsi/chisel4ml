# Chisel4ml
Chisel4ml is an open-source library for generating dataflow architectures inspired by the hls4ml library.


Run "sbt assembly" to build a standalone .jar executable.

To update the protobuffer generated descriptions run:
    protoc -I=. --python_out=./lbir-python LBIR.proto and
    scalapbc --scala_out=flat_package:./scala/ lbir.proto

in the root/chisel4ml directory.
