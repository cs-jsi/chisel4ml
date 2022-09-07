/*
 * TODO
 *
 *
 */

package chisel4ml

import chisel3._
import _root_.chisel4ml.util.bus.AXIStream
import _root_.lbir.{Layer}

class ProcessingElementSequential(layer: Layer, inputDataWidth: Int = 32, outputDataWidth: Int = 32) 
extends Module {
    val io = IO(new Bundle {
        val inStream = Flipped(new AXIStream(inputDataWidth))
        val outStream = new AXIStream(outputDataWidth)
    })

    io.outStream <> io.inStream
}

