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
import _root_.chisel4ml.ProcessingElementSimple
import _root_.chisel4ml.implicits._
import _root_.lbir.{DenseConfig, LayerWrap, Model}
import _root_.scala.collection.mutable._
import _root_.services.GenerateCircuitParams.Options
import _root_.lbir.QTensor

class LBIRStreamSimpleIO(input: QTensor, output: QTensor) extends Bundle {
  val in = Input(Vec(input.width, UInt(input.dtype.bitwidth.W)))
  val out = Output(Vec(output.width, UInt(output.dtype.bitwidth.W)))
}

class ProcessingPipelineSimple(model: Model, options: Options) extends Module with LBIRStreamSimple {
  def layerGeneratorSimple(layer: LayerWrap): Module with LBIRStreamSimple = {
    layer match {
      case l: DenseConfig => Module(ProcessingElementSimple(l))
      case _ => throw new RuntimeException(f"Unsupported layer type")
    }
  }

  // List of processing elements - one PE per layer
  val peList = new ListBuffer[Module with LBIRStreamSimple]()

  // Instantiate modules for seperate layers, for now we only support DENSE layers
  for (layer <- model.layers) {
    peList += layerGeneratorSimple(layer.get)
  }

  val in = IO(Input(Vec(model.layers.head.get.input.width, UInt(model.layers.head.get.input.dtype.bitwidth.W))))
  val out = IO(Output(Vec(model.layers.last.get.output.width, UInt(model.layers.last.get.output.dtype.bitwidth.W))))

  // Connect the inputs and outputs of the layers
  peList(0).in := in
  for (i <- 1 until model.layers.length) {
    if (options.pipelineCircuit) {
      peList(i).in := RegNext(peList(i - 1).out)
    } else {
      peList(i).in := peList(i - 1).out
    }
  }
  out := peList.last.out
}
