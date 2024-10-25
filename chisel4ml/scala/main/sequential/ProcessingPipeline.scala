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
import services.Accelerator

class ProcessingPipeline(accelerators: Seq[Accelerator]) extends Module with HasAXIStream {
  // Instantiate modules for separate layers
  val peList: Seq[Module with HasAXIStream] = accelerators.map(AcceleratorGenerator(_))

  val inStream = IO(chiselTypeOf(peList.head.inStream))
  val outStream = IO(chiselTypeOf(peList.last.outStream))

  // Connect the inputs and outputs of the layers
  peList.head.inStream <> inStream
  for (i <- 1 until accelerators.length) {
    peList(i).inStream <> peList(i - 1).outStream
  }
  outStream <> peList.last.outStream
}
