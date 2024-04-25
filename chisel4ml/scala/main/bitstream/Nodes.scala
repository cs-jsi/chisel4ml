package chisel4ml.bitstream

import chisel3._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.cde.config.Parameters
import chisel3.experimental.SourceInfo

class BSImp[T <: Bits]
    extends SimpleNodeImp[BSMasterParameters[T], BSSlaveParameters[T], BSEdgeParameters[T], BSBundle[T]] {
  def edge(pd: BSMasterParameters[T], pu: BSSlaveParameters[T], p: Parameters, sourceInfo: SourceInfo) = {
    require(pd.bundleParams.genT.getClass == pu.bundleParams.genT.getClass)
    require(pd.bundleParams.genT.getWidth == pu.bundleParams.genT.getWidth)
    require(pd.bundleParams.numBeats == pu.bundleParams.numBeats)
    BSEdgeParameters[T](pd, pu)
  }

  def bundle(e: BSEdgeParameters[T]) = new BSBundle[T](e.master.bundleParams)

  def render(e: BSEdgeParameters[T]) = {
    val nb = e.master.bundleParams.numBeats
    val gt = e.master.bundleParams.genT.toString()
    RenderedEdge(colour = "#0011cc", s"bitstream(numBeats=$nb, genT=$gt)")
  }
}

case class BSMasterNode[T <: Bits](portParams: BSMasterParameters[T])(implicit valName: ValName)
    extends SourceNode(new BSImp[T])(Seq(portParams))
case class BSSlaveNode[T <: Bits](portParams: BSSlaveParameters[T])(implicit valName: ValName)
    extends SinkNode(new BSImp[T])(Seq(portParams))
