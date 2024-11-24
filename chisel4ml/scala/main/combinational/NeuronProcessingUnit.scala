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
package chisel4ml.combinational

import chisel3._
import chisel4ml.compute.NeuronCompute
import chisel4ml.implicits._
import chisel4ml.{HasSimpleStream, LayerMapping}
import lbir.{HasInputOutputQTensor, IsActiveLayer, LayerWrap}

class NeuronProcessingUnit(
  val nc:    NeuronCompute
)(layer:     LayerWrap with HasInputOutputQTensor with IsActiveLayer,
  operation: NeuronOperation)
    extends Module
    with HasSimpleStream {
  val in = IO(Input(Vec(layer.input.numParams, nc.genI)))
  val out = IO(Output(Vec(layer.output.numParams, nc.genO)))

  val kernel: Seq[nc.W] = LayerMapping.getKernel[nc.W](layer)
  val thresh: Seq[nc.A] = LayerMapping.getThresh[nc.A](layer)
  val shift:  Seq[Int] = layer.kernel.dtype.shift

  val inMap:     Seq[Seq[Int]] = LayerMapping.layerToInputMap(layer)
  val kernelMap: Seq[Seq[Int]] = LayerMapping.layerToKernelMap(layer)
  val threshMap: Seq[Int] = LayerMapping.layerToThreshMap(layer)
  val shiftMap:  Seq[Int] = LayerMapping.layerToShiftMap(layer)

  for (i <- 0 until layer.output.numParams) {
    out(i) := operation(nc)(
      LayerMapping.getReceptiveField[nc.I](in.map(_.asInstanceOf[nc.I]), inMap(i)),
      LayerMapping.getReceptiveField[nc.W](kernel, kernelMap(i)),
      thresh(threshMap(i)),
      shift(shiftMap(i))
    )
  }
}
