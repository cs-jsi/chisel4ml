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
import _root_.scala.math

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

class ProcessingElementWrapSimpleToSequential(layer: Layer, options: Options) 
extends ProcessingElementSequential(layer, options) {
    val logger = LoggerFactory.getLogger(classOf[ProcessingElementWrapSimpleToSequential])

    // Input data register
    val inReg = RegInit(VecInit(Seq.fill(numInTrans)(0.U(inputStreamWidth.W)))) 
    val inCntReg = RegInit(0.U((log2(numInTrans) + 1).W))
    val inRegFull = inCntReg === numInTrans.U

    // Output data register
    logger.info(s"""Creating new ProcessingElementWrapSimpleToSequential with inSizeBits: $inSizeBits,
                    | numInTrans: $numInTrans, outSizeBits: $outSizeBits, numOutTrans: $numOutTrans
                    .""".stripMargin.replaceAll("\n",""))
    val outReg = RegInit(VecInit(Seq.fill(numOutTrans)(0.U(outputStreamWidth.W))))
    val outRegUInt = Wire(UInt((numOutTrans*outputStreamWidth).W))
    val outCntReg = RegInit(0.U((log2(numOutTrans) + 1).W))
    val outRegFullReg = RegInit(false.B)

    // (combinational) computational module
    val peSimple = Module(ProcessingElementSimple(layer))
    

    /***** INPUT DATA INTERFACE *****/
    io.inStream.data.ready := !inRegFull
    when(!inRegFull && io.inStream.data.valid) {
        inReg(inCntReg) := io.inStream.data.bits
        inCntReg := inCntReg + 1.U
    }


    /***** OUTPUT DATA INTERFACE *****/
    io.outStream.data.valid := outRegFullReg
    io.outStream.data.bits := outReg(outCntReg)
    io.outStream.last := false.B
    when (outRegFullReg) {
        outCntReg := outCntReg + 1.U
        when ((outCntReg + 1.U) === numOutTrans.U) {
            outRegFullReg := false.B
            io.outStream.last := true.B
        }
    }


    /***** CONNECT INPUT AND OUTPUT REGSITERS WITH THE PE *****/
    peSimple.io.in := inReg.asUInt

    // write data to the output registers
    for (i <- 0 until numOutTrans) outReg(i) := outRegUInt((i+1)*outputStreamWidth - 1, i*outputStreamWidth)
    when(inRegFull && !outRegFullReg) {
        if (outRegUInt.getWidth - outSizeBits == 0) {
            outRegUInt :=  peSimple.io.out
        } else {
            outRegUInt := 0.U((outRegUInt.getWidth - outSizeBits).W) ## peSimple.io.out
        }
        outRegFullReg := true.B
        inCntReg := 0.U
    } .otherwise {
        outRegUInt := 0.U
    }
}
