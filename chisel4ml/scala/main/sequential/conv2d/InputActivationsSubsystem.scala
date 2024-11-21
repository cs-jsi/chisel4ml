package chisel4ml.sequential

import chisel3._
import chisel3.util._
import chisel4ml.HasAXIStreamParameters
import chisel4ml.implicits._
import interfaces.amba.axis.AXIStream
import memories.MemoryGenerator
import org.chipsalliance.cde.config.Parameters

/* InputActivationSubsystem
 * Handles the input data stream, and stores it in to a input buffer. It also "rolls" through the input activation
 * as a convolution opperation would; and does so continously until the next signal is asserted. This allows looping
 * through the input to convolve it with more than one kernel.
 */
class InputActivationsSubsystem[I <: Bits](
  implicit val p: Parameters)
    extends Module
    with HasSequentialConvParameters
    with HasAXIStreamParameters {
  val io = IO(new Bundle {
    val inStream = Flipped(AXIStream(cfg.input.getType[I], numBeatsIn))
    val inputActivationsWindow = Decoupled(Vec(cfg.kernel.numActiveParams(cfg.depthwise), cfg.input.getType[I]))
    val activeDone = Output(Bool())
  })
  val actMem = Module(
    MemoryGenerator.SRAM(
      depth = cfg.input.memDepth(cfg.input.transactionWidth(numBeatsIn)),
      width = cfg.input.transactionWidth(numBeatsIn)
    )
  )
  val dataMover = Module(new InputDataMover[I]())
  val shiftRegConvolver = Module(new ShiftRegisterConvolver[I](cfg))

  object InSubState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sRECEVING_DATA = Value(1.U)
    val sFULL = Value(2.U)
  }
  val state = RegInit(InSubState.sEMPTY)

  val (channelCounterShift, channelCounterWrap) = Counter(0 until cfg.kernel.numChannels, io.activeDone)
  val (kernelCounterShift, _) = Counter(0 until cfg.kernel.numKernels, channelCounterWrap)
  val isLastActiveWindow =
    kernelCounterShift === (cfg.kernel.numKernels - 1).U && channelCounterShift === (cfg.kernel.numChannels - 1).U

  val (actMemCounter, _) = Counter(
    0 to cfg.input.numTransactions(numBeatsIn),
    io.inStream.fire,
    state === InSubState.sFULL && isLastActiveWindow && io.activeDone
  )

  /* INPUT STREAM LOGIC*/
  io.inStream.ready := state =/= InSubState.sFULL
  actMem.io.write.get.address := actMemCounter
  actMem.io.write.get.data := io.inStream.bits.asUInt
  actMem.io.write.get.enable := io.inStream.fire

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
  }.elsewhen(
    state === InSubState.sRECEVING_DATA && actMemCounter === (cfg.input
      .memDepth(cfg.input.transactionWidth(numBeatsIn)) - 1).U && io.inStream.fire
  ) {
    assert(io.inStream.last)
    state := InSubState.sFULL
  }.elsewhen(state === InSubState.sFULL && isLastActiveWindow && io.activeDone) {
    state := InSubState.sEMPTY
  }
}
