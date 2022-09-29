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
import _root_.services.GenerateCircuitParams.Options
import _root_.scala.collection.mutable._


class ProcessingPipeline(model: Model, options: Options) extends Module {
    // List of processing elements - one PE per layer
    val peList = new ListBuffer[ProcessingElementSequential]()

    // Instantiate modules for seperate layers, for now we only support DENSE layers
    for (layer <- model.layers) {
        peList += Module(ProcessingElementSequential(layer, options))
    }

    val io = IO(new Bundle {
        val inStream = Flipped(new AXIStream(32))
        val outStream = new AXIStream(32)
    }) 
    
    // Connect the inputs and outputs of the layers
    peList(0).io.inStream <> io.inStream
    for (i <- 1 until model.layers.length) {
        peList(i).io.inStream <> peList(i - 1).io.outStream
    }
    io.outStream <> peList.last.io.outStream
}
