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
import chisel4ml.compute.OrderCompute
import chisel4ml.implicits._
import chisel4ml.{HasSimpleStream, LayerMapping}
import lbir.{HasInputOutputQTensor, LayerWrap}

class OrderProcessingUnit(
  val oc:    OrderCompute
)(layer:     LayerWrap with HasInputOutputQTensor,
  operation: OrderOperation)
    extends Module
    with HasSimpleStream {
  val in = IO(Input(Vec(layer.input.numParams, layer.input.getType[oc.T])))
  val out = IO(Output(Vec(layer.output.numParams, layer.output.getType[oc.T])))

  val inMap: Seq[Seq[Int]] = LayerMapping.layerToInputMap(layer)

  for (i <- 0 until layer.output.numParams) {
    out(i) := operation(oc)(LayerMapping.getReceptiveField[oc.T](in.map(_.asInstanceOf[oc.T]), inMap(i)))
  }
}
