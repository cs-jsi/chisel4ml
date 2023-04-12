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
import _root_.lbir.{Model, Layer}
import _root_.chisel4ml.implicits._
import _root_.chisel4ml.util.LbirUtil
import _root_.chisel4ml.util.bus.AXIStream
import _root_.services.GenerateCircuitParams.Options
import _root_.scala.collection.mutable._

class ProcessingPipelineSimple(model: Model, options: Options) extends Module {
    // List of processing elements - one PE per layer
    val peList = new ListBuffer[ProcessingElementSimple]()

    // Instantiate modules for seperate layers, for now we only support DENSE layers
    for (layer <- model.layers) {
        peList += Module(ProcessingElementSimple(layer))
    }

    val io = IO(new Bundle {
        val in  = Input(UInt(model.layers.head.input.get.totalBitwidth.W))
        val out = Output(UInt(model.layers.last.output.get.totalBitwidth.W))
    })

    // Connect the inputs and outputs of the layers
    peList(0).io.in := io.in
    for (i <- 1 until model.layers.length) {
        if (options.pipelineCircuit) {
            peList(i).io.in := RegNext(peList(i - 1).io.out)
        } else {
            peList(i).io.in := peList(i - 1).io.out
        }
    }
    io.out := peList.last.io.out
}
