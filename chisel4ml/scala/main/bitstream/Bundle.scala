package chisel4ml.bitstream

import chisel3._
import chisel3.util._

/**
  * BSBundle
  *
  * BSBundle or Bitstream bundle is a small extension to the ReadyValidIO with just the last signal added.
  * The semantics of the last signal mirror that of the last signal in AXIStream.
  *
  * @param params Parameters of the bundle.
  */
class BSBundle[T <: Bits](val params: BSBundleParameters[T])
    extends ReadyValidIO[Vec[T]](Vec(params.numBeats.get, params.genT)) {
  val last = Output(Bool())
}
