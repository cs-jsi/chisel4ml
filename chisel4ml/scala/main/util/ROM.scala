/*
 * SYNTHESIZES TO BLOCK RAM (or several block rams, depending on the depth)
 * (tested in vivado 2021.1)
 */
package chisel4ml.util
import _root_.chisel3._
import _root_.chisel3.util.experimental.loadMemoryFromFileInline 
import _root_.chisel4ml.util.LbirUtil.log2

import _root_.java.io.File
import _root_.java.io.PrintWriter

class ROM(depth: Int, width: Int = 32, memFile:String) extends Module {
    val io = IO(new Bundle {
        val rdEna = Input(Bool())
        val rdAddr = Input(UInt(log2(depth).W))
        val rdData = Output(UInt(width.W))
    })
    val mem = SyncReadMem(depth, UInt(width.W))
    io.rdData := mem.read(io.rdAddr, io.rdEna)
    
    loadMemoryFromFileInline(mem, memFile)
}

