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
import _root_.chisel4ml.util.bus.AXIStream
import _root_.scala.collection.mutable._

class ProcessingPipeline(model: Model, inputDataWidth: Int = 32, outputDataWidth: Int = 32) extends Module {
    // List of processing elements - one PE per layer
    val peList = new ListBuffer[ProcessingElementSequential]()

    // Instantiate modules for seperate layers, for now we only support DENSE layers
    for (layer <- model.layers) {
        peList += Module(new ProcessingElementSequential(layer))
    }

    val io = IO(new Bundle {
        val inStream = Flipped(new AXIStream(inputDataWidth))
        val outStream = new AXIStream(outputDataWidth)
    }) 
    
    // Connect the inputs and outputs of the layers
    peList(0).io.inStream <> io.inStream
    for (i <- 1 until model.layers.length) {
        peList(i).io.inStream <> peList(i - 1).io.outStream
    }
    io.outStream <> peList.last.io.outStream
}
