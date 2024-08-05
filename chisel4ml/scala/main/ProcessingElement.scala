package chisel4ml
import chisel3._
import chisel4ml.bitstream._
import freechips.rocketchip.diplomacy._

trait ProcessingElement[I <: Bits, O <: Bits] extends LazyModule {
  val sNode: BSSlaveNode[I]
  val mNode: BSMasterNode[O]
}
