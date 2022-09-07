/*
 *
 *
 */

package chisel4ml.util.bus

import chisel3._


class AXIStream(val dataWidth : Int) extends Bundle{
  val tdata = Output(UInt(dataWidth.W))
  val tvalid = Output(Bool())
  val tready = Input(Bool())
  val tlast = Output(Bool())
}
