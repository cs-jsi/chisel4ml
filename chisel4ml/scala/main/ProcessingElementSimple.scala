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

import chisel4ml.Neuron
import chisel4ml.implicits._
import chisel4ml._
import lbir.Datatype.QuantizationType._
import lbir.DenseConfig
import org.slf4j.LoggerFactory
import chisel3._

object ProcessingElementSimple {
  def apply(layer: DenseConfig) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.weights.dtype.quantization,
    layer.output.dtype.signed
  ) match {
    case (UNIFORM, true, UNIFORM, false) =>
      new ProcessingElementSimple[SInt, SInt, SInt, SInt, UInt](layer)(UniformQuantizationContextSSUReLU)
    case (UNIFORM, false, UNIFORM, false) =>
      new ProcessingElementSimple[UInt, SInt, SInt, SInt, UInt](layer)(UniformQuantizationContextUSUReLU)
    case (UNIFORM, true, UNIFORM, true) =>
      new ProcessingElementSimple[SInt, SInt, SInt, SInt, SInt](layer)(UniformQuantizationContextSSSNoAct)
    case (UNIFORM, false, UNIFORM, true) =>
      new ProcessingElementSimple[UInt, SInt, SInt, SInt, SInt](layer)(UniformQuantizationContextUSSNoAct)
    case (UNIFORM, false, BINARY, true) =>
      new ProcessingElementSimple[UInt, Bool, SInt, SInt, Bool](layer)(BinaryQuantizationContext)
    case (UNIFORM, true, BINARY, true) =>
      new ProcessingElementSimple[SInt, Bool, SInt, SInt, Bool](layer)(BinaryQuantizationComputeS)
    case (BINARY, _, BINARY, true) =>
      new ProcessingElementSimple[Bool, Bool, Bool, UInt, Bool](layer)(BinarizedQuantizationContext)
    case _ => throw new RuntimeException()
  }
}

class ProcessingElementSimple[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  layer: DenseConfig
)(qc:    QuantizationContext[I, W, M, A, O])
    extends Module
    with LBIRStreamSimple {
  val logger = LoggerFactory.getLogger("ProcessingElementSimple")
  val in = IO(Input(Vec(layer.input.width, layer.input.getType[I])))
  val out = IO(Output(Vec(layer.output.width, layer.output.getType[O])))
  logger.info(f"inner type ${layer.output.getType}")

  val weights: Seq[Seq[W]] = layer.getWeights[W]
  val shift:   Seq[Int] = layer.weights.dtype.shift
  val thresh:  Seq[A] = layer.getThresh[A]

  for (i <- 0 until layer.output.shape(0)) {
    out(i) := Neuron[I, W, M, A, O](
      in.map(_.asInstanceOf[I]),
      weights(i),
      thresh(i),
      shift(i),
      layer.output.dtype.bitwidth
    )(qc)
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
