/*
 * SYNTHESIZES TO BLOCK RAM (or several block rams, depending on the depth)
 * (tested in vivado 2021.1)
 */
package chisel4ml.util
import chisel3._
import _root_.chisel4ml.util.LbirUtil.log2


class SRAM(depth: Int, width: Int = 32) extends Module {
  val io = IO(new Bundle {
    val rdEna = Input(Bool())
    val rdAddr = Input(UInt(log2(depth).W))
    val rdData = Output(UInt(width.W))
    val wrEna = Input(Bool())
    val wrAddr = Input(UInt(log2(depth).W))
    val wrData = Input(UInt(width.W))
  })
  val mem = SyncReadMem(depth, UInt(width.W))

  // Create one write port and one read port
  when (io.wrEna) {
    mem.write(io.wrAddr, io.wrData)
  }
  io.rdData := mem.read(io.rdAddr, io.rdEna)
}

