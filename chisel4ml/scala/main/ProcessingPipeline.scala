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

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.chisel3.experimental._
import _root_.lbir.Model
import _root_.chisel4ml.{LayerGenerator, LBIRStream}
import interfaces.amba.axis._
import _root_.services.GenerateCircuitParams.Options
import _root_.scala.collection.mutable._


class ProcessingPipeline(model: Model, options: Options) extends Module with LBIRStream {
    val inStream = IO(Flipped(AXIStream(UInt(options.layers(0).busWidthIn.W))))
    val outStream = IO(AXIStream(UInt(options.layers.last.busWidthOut.W)))

    // List of processing elements - one PE per layer
    val peList = new ListBuffer[Module with LBIRStream]()

    // Instantiate modules for seperate layers, for now we only support DENSE layers
    for ((layer, idx) <- model.layers.zipWithIndex) {
        peList += LayerGenerator(layer.get, options.layers(idx))
    }

    // Connect the inputs and outputs of the layers
    peList(0).inStream <> inStream
    for (i <- 1 until model.layers.length) {
        peList(i).inStream <> peList(i - 1).outStream
    }
    outStream <> peList.last.outStream
}
