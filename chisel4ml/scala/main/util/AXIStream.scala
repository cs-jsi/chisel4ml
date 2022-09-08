/*
 *
 *
 */

package chisel4ml.util.bus

import chisel3._
import chisel3.util._

class AXIStream(val dataWidth : Int) extends Bundle{
  val data = Irrevocable(Output(UInt(dataWidth.W)))
  val last = Output(Bool())
}
