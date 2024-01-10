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
import lbir.Model
import scala.collection.mutable._
import interfaces.amba.axis._

class ProcessingPipeline(model: Model) extends Module with HasLBIRStream {
  val inStream = IO(Flipped(AXIStream(UInt(model.layers.head.get.input.dtype.bitwidth.W))))
  val outStream = IO(AXIStream(UInt(model.layers.last.get.output.dtype.bitwidth.W))) // TODO

  // List of processing elements - one PE per layer
  val peList = new ListBuffer[Module with HasLBIRStream]()

  // Instantiate modules for seperate layers
  for ((layer, idx) <- model.layers.zipWithIndex) {
    peList += LayerGenerator(layer.get)
  }

  // Connect the inputs and outputs of the layers
  peList.head.inStream <> inStream
  for (i <- 1 until model.layers.length) {
    peList(i).inStream <> peList(i - 1).outStream
  }
  outStream <> peList.last.outStream
}
