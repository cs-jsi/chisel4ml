/*
 * SYNTHESIZES TO BLOCK RAM (or several block rams)
 * (tested in vivado 2021.1)
 */
package chisel4ml.util
import chisel3._
import chisel4ml.LbirUtil.log2


class SRAM(depth: Int, width: Int = 32) extends Module {
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(log2(depth).W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })

  val mem = SyncReadMem(depth, UInt(width.W))
  // Create one write port and one read port
  mem.write(io.addr, io.dataIn)
  io.dataOut := mem.read(io.addr, io.enable)
}

