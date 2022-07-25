/*
 * HEADER: TODO
 *
 *
 * This file contains the definition of the Model.
 */

package chisel4ml

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.chisel3.experimental._
import _root_.lbir.{Model, Layer}
import _root_.chisel4ml.util.LbirUtil 
import _root_.scala.collection.mutable._

class ProcessingPipeline(model: Model) extends Module {
    // List of processing elements - one PE per layer
    val peList = new ListBuffer[ProcessingElementSimple]()

    // Instantiate modules for seperate layers, for now we only support DENSE layers
    for (layer <- model.layers) {
        peList += Module(ProcessingElementSimple(layer))
    }

    val io = IO(new Bundle {
        val in  = Input(UInt(LbirUtil.qtensorTotalBitwidth(model.layers.head.input.get).W))
        val out = Output(UInt(LbirUtil.qtensorTotalBitwidth(model.layers.last.output.get).W))
    }) 

    // Connect the inputs and outputs of the layers
    peList(0).io.in := io.in
    for (i <- 1 until model.layers.length) {
        peList(i).io.in := peList(i - 1).io.out
    }
    io.out := peList.last.io.out
}
