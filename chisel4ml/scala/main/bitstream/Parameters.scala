package chisel4ml.bitstream

import chisel3.Bits

case class BSBundleParameters[T <: Bits](
  genT:     T,
  numBeats: Option[Int])

case class BSMasterParameters[T <: Bits](
  bundleParams: BSBundleParameters[T])

case class BSSlaveParameters[T <: Bits](
  bundleParams: BSBundleParameters[T])

case class BSEdgeParameters[T <: Bits](
  master: BSMasterParameters[T],
  slave:  BSSlaveParameters[T])
