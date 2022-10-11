/*
 * TODO
 *
 *
 */

package chisel4ml

import chisel3._
import chisel3.util._

import _root_.chisel4ml.util.bus.AXIStream
import _root_.chisel4ml.util.SRAM
import _root_.chisel4ml.util.LbirUtil.log2
import _root_.lbir.{Layer}
import _root_.services.GenerateCircuitParams.Options

abstract class ProcessingElementSequential(layer: Layer, options: Options) extends Module {
    val inputStreamWidth = 32
    val outputStreamWidth = 32

    val inSizeBits: Int = layer.input.get.totalBitwidth
    val numInTrans: Int = math.ceil(inSizeBits.toFloat / inputStreamWidth.toFloat).toInt

    val outSizeBits: Int = layer.output.get.totalBitwidth
    val numOutTrans: Int = math.ceil(outSizeBits.toFloat / outputStreamWidth.toFloat).toInt

    val io = IO(new Bundle {
        val inStream = Flipped(new AXIStream(inputStreamWidth))
        val outStream = new AXIStream(outputStreamWidth)
    })
}

object ProcessingElementSequential {
    def apply(layer: Layer, options: Options) = new ProcessingElementWrapSimpleToSequential(layer, options)
}
