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
import chisel4ml.ProcessingElementSimple
import chisel4ml.implicits._
import lbir.{DenseConfig, LayerWrap, Model}
import scala.collection.mutable._
import services.GenerateCircuitParams.Options
import lbir.QTensor

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

  val in = IO(Input(Vec(model.layers.head.get.input.width, model.layers.head.get.input.getType)))
  val out = IO(Output(Vec(model.layers.last.get.output.width, model.layers.last.get.output.getType)))

  // Connect the inputs and outputs of the layers
  peList.head.in := in
  for (i <- 1 until model.layers.length) {
    if (options.pipelineCircuit) {
      peList(i).in := RegNext(peList(i - 1).out)
    } else {
      peList(i).in := peList(i - 1).out
    }
  }
  out := peList.last.out
}
