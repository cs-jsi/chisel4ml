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
  def apply(layer: DenseConfig) = {
    new ProcessingElementSimple(layer)(LayerGenerator.layerToQC(layer))
  }
}

class ProcessingElementSimple[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
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
    out(i) := Neuron[I, W, M, A, O](
      in.map(_.asInstanceOf[I]),
      weights(i),
      thresh(i),
      shift(i),
      layer.output.dtype.bitwidth,
      useThresh = !(layer.kernel.dtype.quantization == BINARY && layer.input.dtype.quantization == BINARY),
      layer.roundingMode
    )(qc)
  }
}
