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

class ProcessingElementSimple(
  layer:  DenseConfig
)(val qc: QuantizationContext)
    extends Module
    with LBIRStreamSimple {
  val in = IO(Input(Vec(layer.input.width, layer.input.getType[qc.I])))
  val out = IO(Output(Vec(layer.output.width, layer.output.getType[qc.O])))

  val weights: Seq[Seq[qc.W]] = layer.getWeights[qc.W]
  val shift:   Seq[Int] = layer.kernel.dtype.shift
  val thresh:  Seq[qc.A] = layer.getThresh[qc.A]

  val Neuron: StaticNeuron =
    if (layer.kernel.dtype.quantization == BINARY && layer.input.dtype.quantization == BINARY) {
      NeuronWithoutBias
    } else {
      NeuronWithBias
    }
  for (i <- 0 until layer.output.shape(0)) {
    out(i) := Neuron(qc)(
      in.map(_.asInstanceOf[qc.I]),
      weights(i),
      thresh(i),
      shift(i)
    )
  }
}
