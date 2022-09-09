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

class ProcessingElementSequential(layer: Layer, inputDataWidth: Int = 32, outputDataWidth: Int = 32) 
extends Module {
    val io = IO(new Bundle {
        val inStream = Flipped(new AXIStream(inputDataWidth))
        val outStream = new AXIStream(outputDataWidth)
    })
    val memSize = 256
    val memSizeWords = memSize / inputDataWidth

    val mem = Module(new SRAM(memSizeWords, inputDataWidth))
    val getDataState :: delayState :: pushDataState :: Nil = Enum(3)
    val stateReg = RegInit(getDataState)
    val cntReg = RegInit(0.U((log2(memSizeWords) + 1).W)) // counts the number of bytes
    //val outStreamDataReg = RegInit(0.U(outputDataWidth.W))
    //val outStreamLastReg = RegInit(false.B)

    // next state logic
    when (stateReg === getDataState) {
        when (io.inStream.last === true.B) {
            stateReg := delayState
        }
    } .elsewhen (stateReg === delayState) {
        stateReg := pushDataState
    } .otherwise { // state === pushDataState
        when (cntReg === memSizeWords.U) {
            stateReg := getDataState
        }
    }

    
    // counter logic
    when (stateReg === getDataState) {
        when (io.inStream.data.valid && io.inStream.data.ready) {
            cntReg := cntReg + 1.U
        }
        when (io.inStream.last === true.B) {
            cntReg := 0.U // Reset for new AXI Stream packet
        }
    } .elsewhen(stateReg === delayState) {
        cntReg := cntReg + 1.U
    } .otherwise { // stateReg === pushDataState
        when (io.outStream.data.valid && io.outStream.data.ready) {
            cntReg := cntReg + 1.U
        }
    }

    
    // memory defaults
    mem.io.enable := false.B
    mem.io.write := false.B
    mem.io.addr := cntReg
    mem.io.dataIn := 0.U

    
    // input stream logic
    io.inStream.data.ready := stateReg === getDataState
    when (io.inStream.data.ready && io.inStream.data.valid) {
        mem.io.write := true.B
        mem.io.dataIn := io.inStream.data.bits
        mem.io.enable := true.B
    }

    
    // output stream logic
    when (stateReg === delayState || stateReg === pushDataState) {
        mem.io.enable := true.B
    }

    io.outStream.data.valid := stateReg === pushDataState
    io.outStream.data.bits := 0.U
    io.outStream.last := false.B
    when (io.outStream.data.ready && io.outStream.data.valid) {
        io.outStream.data.bits := mem.io.dataOut
        when (cntReg === memSizeWords.U) {
            io.outStream.last := true.B
        }
    }
    //io.outStream <> io.inStream
}

