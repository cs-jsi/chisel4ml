package chisel4ml.bitstream

import chisel3._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.cde.config.Parameters
import chisel3.experimental.SourceInfo

class BSImp[T <: Bits]
    extends SimpleNodeImp[BSMasterParameters[T], BSSlaveParameters[T], BSEdgeParameters[T], BSBundle[T]] {
  def edge(pd: BSMasterParameters[T], pu: BSSlaveParameters[T], p: Parameters, sourceInfo: SourceInfo) = {
    require(pd.numBeats.isDefined || pu.numBeats.isDefined)
    require(pd.genT.getClass() == pu.genT.getClass())
    require(pd.genT.getWidth == pu.genT.getWidth)
    // require(pd.tensor == pu.tensor)
    val numBeats = if (pd.numBeats.isDefined) pd.numBeats.get else pu.numBeats.get
    BSEdgeParameters(pd.tensor, pd.genT, numBeats)
  }

  def bundle(e: BSEdgeParameters[T]) = new BSBundle[T](
    BSBundleParameters[T](
      e.genT,
      e.numBeats
    )
  )

  def render(e: BSEdgeParameters[T]) = {
    val nb = e.numBeats
    val gt = e.genT.toString()
    RenderedEdge(colour = "#0011cc", s"bitstream(numBeats=$nb, genT=$gt)")
  }
}

case class BSMasterNode[T <: Bits](portParams: BSMasterParameters[T])(implicit valName: ValName)
    extends SourceNode(new BSImp[T])(Seq(portParams))
case class BSSlaveNode[T <: Bits](portParams: BSSlaveParameters[T])(implicit valName: ValName)
    extends SinkNode(new BSImp[T])(Seq(portParams))
