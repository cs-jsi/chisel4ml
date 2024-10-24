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
import lbir.{HasInputOutputQTensor, LayerWrap}

class ProcessingPipelineSimple(layers: Seq[LayerWrap with HasInputOutputQTensor]) extends Module with LBIRStreamSimple {

  // Instantiate modules for seperate layers
  val peList: Seq[Module with LBIRStreamSimple] = layers.map { l: LayerWrap =>
    AcceleratorGeneratorCombinational(l)
  }
  val in = IO(Input(Vec(layers.head.input.width, layers.head.input.getType)))
  val out = IO(Output(Vec(layers.last.output.width, layers.last.output.getType)))

  // Connect the inputs and outputs of the layers
  peList.head.in := in
  for (i <- 1 until layers.length) {
    peList(i).in := RegNext(peList(i - 1).out)
  }
  out := peList.last.out
}
