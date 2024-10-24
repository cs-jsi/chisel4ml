package chisel4ml

import chisel3._
import chisel3.util._
import chisel4ml.implicits._
import interfaces.amba.axis.AXIStreamIO
import lbir.QTensor

abstract class BufferType
case class RollingBufferType(bufferSize: Int) extends BufferType {
  require(bufferSize > 0)
}
/* LogicalAccessPattern
 *
 * Defines a access pattern into a QTensor.
 */
class LogicalAccessPattern(val pattern: Seq[Int], val qtensor: QTensor)
class LinearLAP(qtensor: QTensor) extends LogicalAccessPattern(0 until qtensor.numParams, qtensor)
class RollingBufferAccessPattern(lap: LogicalAccessPattern, rollingBuffer: RollingBufferType) {
  def qtensor = lap.qtensor
}

trait StreamReader {
  def readToBuffer(
    stream:     AXIStreamIO[_ <: Data],
    qtensor:    QTensor,
    ready:      Bool,
    windowSize: (Int, Int)
  ): (Seq[Data], Bool, Bool)
}

/* RollingBufferReader
 *
 * Reads a stream into a rolling buffer and gets a window of the stream and offers it via a handshake interface.
 *
 */
object RollingBufferReader extends StreamReader {
  override def readToBuffer(
    stream:     AXIStreamIO[_ <: Data],
    qtensor:    QTensor,
    ready:      Bool,
    windowSize: (Int, Int)
  ): (Seq[Data], Bool, Bool) = {
    require(windowSize._1 > 0)
    require(windowSize._2 > 0)
    require(qtensor.layout == "NCHW")

    object InputBufferState extends ChiselEnum {
      val sEMPTY = Value(0.U)
      val sREAD_BEAT = Value(1.U)
      val sSTALL = Value(2.U)
    }
    val state = RegInit(InputBufferState.sEMPTY)
    val inputsBuffer = RegEnable(stream.bits, stream.fire)
    val getNext = stream.fire || state === InputBufferState.sREAD_BEAT
    val (bufferCounter, _) = Counter(getNext, stream.beats)
    val nextData = inputsBuffer(bufferCounter)
    val (_, widthCounterWrap) = Counter(getNext, qtensor.width)
    val (_, heightCounterWrap) = Counter(widthCounterWrap, qtensor.height)
    val (_, channelCounterWrap) = Counter(heightCounterWrap, qtensor.numChannels)
    val bufferSize = qtensor.height * windowSize._1 - (qtensor.width - windowSize._2)
    val rollingBuffer = ShiftRegisters(nextData, bufferSize, ready)
    val outputValid = true.B
    val outputLast = true.B
    val inputReady = true.B
    stream.ready := inputReady
    (rollingBuffer, outputValid, outputLast)
  }

}
