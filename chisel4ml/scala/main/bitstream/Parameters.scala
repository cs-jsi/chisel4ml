package chisel4ml.bitstream

import chisel3.Bits
import lbir.QTensor

case class BSBundleParameters[T <: Bits](
  genT:     T,
  numBeats: Int)

case class BSMasterParameters[T <: Bits](
  tensor:   QTensor,
  genT:     T,
  numBeats: Option[Int])

case class BSSlaveParameters[T <: Bits](
  tensor:   QTensor,
  genT:     T,
  numBeats: Option[Int])

case class BSEdgeParameters[T <: Bits](
  tensor:   QTensor,
  genT:     T,
  numBeats: Int) {
  val totalWidth = genT.getWidth * numBeats
}
