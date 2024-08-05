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

import chisel3._
import chisel3.util._
import chisel4ml.implicits._
import chisel4ml.quantization.QuantizationContext
import lbir.Datatype.QuantizationType.BINARY
import lbir.DenseConfig

class ProcessingElementDenseSimple[I <: Bits, W <: Bits, M <: Bits, A <: Bits, O <: Bits](
  l: DenseConfig
)(
  implicit val qc: QuantizationContext[I, W, M, A, O])
    extends Module {
  val in = IO(Input(Valid(Vec(l.input.width, l.input.getType[I]))))
  val out = IO(Output(Valid(Vec(l.output.width, l.output.getType[O]))))
  out.valid := in.valid

  val weights: Seq[Seq[W]] = l.getWeights[W]
  val shift:   Seq[Int] = l.kernel.dtype.shift
  val thresh:  Seq[A] = l.getThresh[A]

  for (i <- 0 until l.output.shape(0)) {
    out.bits(i) := Neuron[I, W, M, A, O](
      in.bits.map(_.asInstanceOf[I]),
      weights(i),
      thresh(i),
      shift(i),
      l.output.dtype.bitwidth,
      useThresh = !(l.kernel.dtype.quantization == BINARY && l.input.dtype.quantization == BINARY),
      roundingMode = l.roundingMode
    )
  }
}
