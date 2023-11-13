/*
 * Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package chisel4ml

import chisel4ml.combinational.Neuron
import chisel4ml.implicits._
import chisel4ml.util._
import chisel4ml._
import lbir.Activation.{BINARY_SIGN, NO_ACTIVATION, RELU}
import lbir.Datatype.QuantizationType._
import lbir.DenseConfig
import org.slf4j.LoggerFactory
import chisel3._
import chisel3.util._
import dsptools.DspException

object ProcessingElementSimple {
  def apply(layer: DenseConfig) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.weights.dtype.quantization
  ) match {
    case (UNIFORM, true, UNIFORM) =>
      new ProcessingElementSimple[SInt, SInt, SInt, SInt, UInt](layer)(UniformQuantizationComputeSSUReLU)
    case (UNIFORM, false, UNIFORM) =>
      new ProcessingElementSimple[UInt, SInt, SInt, SInt, UInt](layer)(UniformQuantizationComputeUSUReLU)
    case (UNIFORM, false, BINARY) =>
      new ProcessingElementSimple[UInt, Bool, SInt, SInt, Bool](layer)(BinaryQuantizationCompute)
    case (UNIFORM, true, BINARY) =>
      new ProcessingElementSimple[SInt, Bool, SInt, SInt, Bool](layer)(BinaryQuantizationComputeS)
    case (BINARY, _, BINARY) =>
      new ProcessingElementSimple[Bool, Bool, Bool, UInt, Bool](layer)(BinarizedQuantizationCompute)
    case _ => throw new RuntimeException()
  }
}

class ProcessingElementSimple[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  layer: DenseConfig
)(qc:    QuantizationCompute[I, W, M, A, O])
    extends Module
    with LBIRStreamSimple {
  val logger = LoggerFactory.getLogger("ProcessingElementSimple")
  val in = IO(Input(Vec(layer.input.width, layer.input.getType)))
  val out = IO(Output(Vec(layer.output.width, layer.output.getType)))
  val weights: Seq[Seq[W]] = layer.getWeights[W]
  val thresh:  Seq[A] = layer.getThresh[A]
  val shift:   Seq[Int] = layer.weights.dtype.shift

  for (i <- 0 until layer.output.shape(0)) {
    out(i) := Neuron[I, W, M, A, O](in.map(_.asInstanceOf[I]), weights(i), thresh(i), shift(i))(qc)
  }

  logger.info(
    s"""Created new ProcessingElementSimpleDense processing element. It has an input shape:
       | ${layer.input.shape} and output shape: ${layer.output.shape}. The input bitwidth
       | is ${layer.input.dtype.bitwidth}, the output bitwidth
       | ${layer.output.dtype.bitwidth}. Thus the total size of the input vector is
       | ${layer.input.totalBitwidth} bits, and the total size of the output vector
       | is ${layer.output.totalBitwidth} bits.
       | The input quantization is ${layer.input.getType}, output quantization is ${layer.output.getType}.""".stripMargin
      .replaceAll("\n", "")
  )
}
