/*
 * TODO
 *
 *
 */

package chisel4ml

import chisel3._
import chisel3.util._

import _root_.chisel4ml.util.bus.AXIStream
import _root_.chisel4ml.util.{SRAM, ROM}
import _root_.chisel4ml.util.LbirUtil.log2
import _root_.chisel4ml.util.LbirUtil
import _root_.lbir.{Layer}
import _root_.services.GenerateCircuitParams.Options
import _root_.scala.math


class ProcessingElementSequentialConv(layer: Layer, options: Options) 
extends ProcessingElementSequential(layer, options) {                    
    val inReg = RegInit(0.U(inputStreamWidth.W))
    
    val sramMemDepth = 4
    val sram = Module(new SRAM(depth=sramMemDepth, width=32))
    val sramAddr = RegInit(0.U((log2(sramMemDepth) + 1).W))

    val romMemDepth = 4
    val rom = Module(new ROM(depth=romMemDepth, width=32, memFile=LbirUtil.createHexMemoryFile(layer.weights.get)))
    val romAddr = RegInit(0.U((log2(romMemDepth) + 1).W))

    /***** INPUT DATA INTERFACE *****/
    io.inStream.data.ready := sramAddr < sramMemDepth.U
    when(io.inStream.data.ready && io.inStream.data.valid) {
        inReg := io.inStream.data.bits
        sramAddr := sramAddr + 1.U
    }

    // Handles the SRAM memory
    val wasInputTrans = RegNext(io.inStream.data.ready && io.inStream.data.valid)
    sram.io.wrEna := false.B
    sram.io.wrAddr := 0.U
    sram.io.wrData := inReg
    sram.io.rdEna := false.B
    sram.io.rdAddr := 0.U
    when(wasInputTrans) {
        sram.io.wrEna := true.B
        sram.io.wrAddr := sramAddr
    }


    /***** OUTPUT DATA INTERFACE *****/
    io.outStream.data.valid := (romAddr > 0.U) && (romAddr <= romMemDepth.U)
    io.outStream.data.bits := rom.io.rdData
    io.outStream.last := romAddr === (romMemDepth - 1).U

    // Handle ROM memory
    rom.io.rdEna := true.B
    rom.io.rdAddr := romAddr

    // Addr counter
    when(romAddr <= romMemDepth.U) {
        romAddr := romAddr + 1.U
    }
}
