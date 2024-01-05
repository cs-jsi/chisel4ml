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
    val inputActivationsWindow = Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), l.input.getType[I]))
    val activeDone = Output(Bool())
  })
  val actMem = Module(MemoryGenerator.SRAM(depth = l.input.memDepth(), width = MemWordSize.bits))
  val dataMover = Module(new InputDataMover(l.input))
  val shiftRegConvolver = Module(new ShiftRegisterConvolver(l))

  object InSubState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sRECEVING_DATA = Value(1.U)
    val sFULL = Value(2.U)
  }
  val state = RegInit(InSubState.sEMPTY)

  val (channelCounterShift, channelCounterWrap) = Counter(0 until l.kernel.numChannels, io.activeDone)
  val (kernelCounterShift, _) = Counter(0 until l.kernel.numKernels, channelCounterWrap)
  val isLastActiveWindow =
    kernelCounterShift === (l.kernel.numKernels - 1).U && channelCounterShift === (l.kernel.numChannels - 1).U

  val (actMemCounter, _) = Counter(
    0 to l.input.numTransactions(options.busWidthIn),
    io.inStream.fire,
    state === InSubState.sFULL && isLastActiveWindow && io.activeDone
  )

  /* INPUT STREAM LOGIC*/
  io.inStream.ready := state =/= InSubState.sFULL
  actMem.io.write.address := actMemCounter
  actMem.io.write.data := io.inStream.bits
  actMem.io.write.enable := io.inStream.fire

  dataMover.io.actMem <> actMem.io.read
  dataMover.io.actMemWrittenTo := actMemCounter

  // Start one cycle after start of transmission of the input packet or if already loaded in next cycle
  val startOfTransmission = state === InSubState.sRECEVING_DATA && RegNext(state === InSubState.sEMPTY)
  dataMover.io.start := RegNext(startOfTransmission) ||
    (RegNext(io.activeDone && !isLastActiveWindow && channelCounterWrap))

  shiftRegConvolver.io.nextElement <> dataMover.io.nextElement
  io.inputActivationsWindow <> shiftRegConvolver.io.inputActivationsWindow
  io.activeDone := shiftRegConvolver.io.channelDone

  when(state === InSubState.sEMPTY && io.inStream.fire) {
    state := InSubState.sRECEVING_DATA
  }.elsewhen(state === InSubState.sRECEVING_DATA && actMemCounter === (l.input.memDepth() - 1).U && io.inStream.fire) {
    assert(io.inStream.last)
    state := InSubState.sFULL
  }.elsewhen(state === InSubState.sFULL && isLastActiveWindow && io.activeDone) {
    state := InSubState.sEMPTY
  }
}
