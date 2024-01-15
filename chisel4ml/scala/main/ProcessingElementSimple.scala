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

import chisel4ml.{NeuronWithBias, NeuronWithoutBias}
import chisel4ml.implicits._
import chisel4ml.quantization._
import chisel4ml._
import lbir.Datatype.QuantizationType._
import lbir.DenseConfig
import chisel3._
import spire.algebra.Ring
import spire.implicits._
import dsptools.numbers._

object ProcessingElementSimple {
  def apply(layer: DenseConfig) = (
    layer.input.dtype.quantization,
    layer.input.dtype.signed,
    layer.kernel.dtype.quantization,
    layer.output.dtype.signed
  ) match {
    case (UNIFORM, true, UNIFORM, false) =>
      new ProcessingElementSimple[SInt, SInt, SInt, SInt, UInt](layer)(
        new UniformQuantizationContextSSUReLU(layer.roundingMode)
      )
    case (UNIFORM, false, UNIFORM, false) =>
      new ProcessingElementSimple[UInt, SInt, SInt, SInt, UInt](layer)(
        new UniformQuantizationContextUSUReLU(layer.roundingMode)
      )
    case (UNIFORM, true, UNIFORM, true) =>
      new ProcessingElementSimple[SInt, SInt, SInt, SInt, SInt](layer)(
        new UniformQuantizationContextSSSNoAct(layer.roundingMode)
      )
    case (UNIFORM, false, UNIFORM, true) =>
      new ProcessingElementSimple[UInt, SInt, SInt, SInt, SInt](layer)(
        new UniformQuantizationContextUSSNoAct(layer.roundingMode)
      )
    case (UNIFORM, false, BINARY, true) =>
      new ProcessingElementSimple[UInt, Bool, SInt, SInt, Bool](layer)(
        new BinaryQuantizationContext(layer.roundingMode)
      )
    case (UNIFORM, true, BINARY, true) =>
      new ProcessingElementSimple[SInt, Bool, SInt, SInt, Bool](layer)(
        new BinaryQuantizationContextSInt(layer.roundingMode)
      )
    case (BINARY, _, BINARY, true) =>
      new ProcessingElementSimple[Bool, Bool, Bool, UInt, Bool](layer)(BinarizedQuantizationContext)
    case _ => throw new RuntimeException()
  }
}

class ProcessingElementSimple[I <: Bits, W <: Bits, M <: Bits, A <: Bits: Ring, O <: Bits](
  layer: DenseConfig
)(qc:    QuantizationContext[I, W, M, A, O])
    extends Module
    with LBIRStreamSimple {
  val in = IO(Input(Vec(layer.input.width, layer.input.getType[I])))
  val out = IO(Output(Vec(layer.output.width, layer.output.getType[O])))

  val weights: Seq[Seq[W]] = layer.getWeights[W]
  val shift:   Seq[Int] = layer.kernel.dtype.shift
  val thresh:  Seq[A] = layer.getThresh[A]

  for (i <- 0 until layer.output.shape(0)) {
    if (layer.kernel.dtype.quantization == BINARY && layer.input.dtype.quantization == BINARY) {
      out(i) := NeuronWithoutBias[I, W, M, A, O](
        in.map(_.asInstanceOf[I]),
        weights(i),
        thresh(i),
        shift(i),
        layer.output.dtype.bitwidth
      )(qc)
    } else {
      out(i) := NeuronWithBias[I, W, M, A, O](
        in.map(_.asInstanceOf[I]),
        weights(i),
        thresh(i),
        shift(i),
        layer.output.dtype.bitwidth
      )(qc)
    }
  }
}
