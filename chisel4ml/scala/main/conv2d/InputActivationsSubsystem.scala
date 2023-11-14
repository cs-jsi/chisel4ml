package chisel4ml.conv2d

import chisel3._
import chisel3.util._
import interfaces.amba.axis.AXIStream
import services.LayerOptions
import memories.MemoryGenerator
import chisel4ml.MemWordSize
import chisel4ml.implicits._
import lbir.Conv2DConfig

/* InputActivationSubsystem
 * Handles the input data stream, and stores it in to a input buffer. It also "rolls" through the input activation
 * as a convolution opperation would; and does so continously until the next signal is asserted. This allows looping
 * through the input to convolve it with more than one kernel.
 */
class InputActivationsSubsystem[I <: Bits](l: Conv2DConfig, options: LayerOptions) extends Module {
  val io = IO(new Bundle {
    val inStream = Flipped(AXIStream(UInt(options.busWidthIn.W)))
    val inputActivationsWindow = Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), l.input.getType.asInstanceOf[I]))
    val activeDone = Output(Bool())
  })
  val actMem = Module(MemoryGenerator.SRAM(depth = l.input.memDepth, width = MemWordSize.bits))
  val dataMover = Module(new InputDataMover(l.input))
  val shiftRegConvolver = Module(new ShiftRegisterConvolver(l))

  val (_, chCntWrap) = Counter(0 until l.kernel.numKernels, dataMover.io.done)

  object InSubState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sRECEVING_DATA = Value(1.U)
    val sFULL = Value(2.U)
  }
  val state = RegInit(InSubState.sEMPTY)

  /* INPUT STREAM LOGIC*/
  val (actMemCntValue, _) = Counter(0 to l.input.memDepth, io.inStream.fire, chCntWrap)
  io.inStream.ready := state =/= InSubState.sFULL
  actMem.io.write.address := actMemCntValue
  actMem.io.write.data := io.inStream.bits
  actMem.io.write.enable := io.inStream.fire

  dataMover.io.actMem <> actMem.io.read
  dataMover.io.actMemWrittenTo := actMemCntValue

  // Start one cycle after start of transmission of the input packet or if already loaded in next cycle
  val startOfTransmission = state === InSubState.sRECEVING_DATA && RegNext(state === InSubState.sEMPTY)
  dataMover.io.start := RegNext(startOfTransmission) || RegNext(dataMover.io.done && !chCntWrap)

  shiftRegConvolver.io.nextElement <> dataMover.io.nextElement
  io.inputActivationsWindow <> shiftRegConvolver.io.inputActivationsWindow
  io.activeDone := shiftRegConvolver.io.channelDone

  when(state === InSubState.sEMPTY && io.inStream.fire) {
    state := InSubState.sRECEVING_DATA
  }.elsewhen(state === InSubState.sRECEVING_DATA && actMemCntValue === (l.input.memDepth - 1).U) {
    state := InSubState.sFULL
  }.otherwise {
    when(dataMover.io.done) {
      when(chCntWrap) { state := InSubState.sEMPTY }.otherwise { state := InSubState.sFULL }
    }
  }
}
