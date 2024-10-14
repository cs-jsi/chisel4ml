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
import chisel4ml.implicits._
import chisel4ml.quantization._
import lbir.{HasInputOutputQTensor, IsActiveLayer, LayerWrap}

class ProcessingElementCombinational(
  val qc:    QuantizationContext
)(layer:     LayerWrap with HasInputOutputQTensor with IsActiveLayer,
  operation: Transformation)
    extends Module
    with LBIRStreamSimple {
  val in = IO(Input(Vec(layer.input.width, layer.input.getType[qc.I])))
  val out = IO(Output(Vec(layer.output.width, layer.output.getType[qc.O])))

  val kernel: Seq[qc.W] = LayerMapping.getKernel[qc.W](layer)
  val thresh: Seq[qc.A] = LayerMapping.getThresh[qc.A](layer)
  val shift:  Seq[Int] = layer.kernel.dtype.shift

  val inMap:     Seq[Seq[Int]] = LayerMapping.layerToInputMap(layer)
  val kernelMap: Seq[Seq[Int]] = LayerMapping.layerToKernelMap(layer)
  val threshMap: Seq[qc.A] = thresh
  val shiftMap:  Seq[Int] = shift

  for (i <- 0 until layer.output.shape(0)) {
    out(i) := operation(qc)(
      LayerMapping.getReceptiveField[qc.I](in.map(_.asInstanceOf[qc.I]), inMap(i)),
      LayerMapping.getReceptiveField[qc.W](kernel, kernelMap(i)),
      threshMap(i),
      shiftMap(i)
    )
  }
}

class ProcessingElementCombinationalIO(
  val qc:      QuantizationContext
)(layer:       LayerWrap with HasInputOutputQTensor,
  operationIO: TransformationIO)
    extends Module
    with LBIRStreamSimple {
  val in = IO(Input(Vec(layer.input.width, layer.input.getType[qc.I])))
  val out = IO(Output(Vec(layer.output.width, layer.output.getType[qc.O])))

  val inMap: Seq[Seq[Int]] = LayerMapping.layerToInputMap(layer)

  for (i <- 0 until layer.output.shape(0)) {
    out(i) := operationIO(qc)(LayerMapping.getReceptiveField[qc.I](in.map(_.asInstanceOf[qc.I]), inMap(i)))
  }
}
