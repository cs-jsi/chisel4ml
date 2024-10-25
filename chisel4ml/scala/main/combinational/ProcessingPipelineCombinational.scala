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
import lbir.{HasInputOutputQTensor, LayerWrap}

class ProcessingPipelineCombinational(layers: Seq[LayerWrap with HasInputOutputQTensor])
    extends Module
    with HasSimpleStream {
  // Instantiate modules for separate layers
  val peList: Seq[Module with HasSimpleStream] = layers.map { l: LayerWrap =>
    AcceleratorGeneratorCombinational(l)
  }
  val in = IO(Input(chiselTypeOf(peList.head.in)))
  val out = IO(Output(chiselTypeOf(peList.last.out)))

  // Connect the inputs and outputs of the layers
  peList.head.in := in
  for (i <- 1 until layers.length) {
    peList(i).in := RegNext(peList(i - 1).out)
  }
  out := peList.last.out
}
