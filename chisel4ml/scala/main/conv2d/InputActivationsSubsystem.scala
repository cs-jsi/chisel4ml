package chisel4ml.conv2d

import chisel3._
import chisel3.util._
import interfaces.amba.axis.AXIStream
import services.LayerOptions
import memories.MemoryGenerator
import chisel4ml.MemWordSize
import chisel4ml.implicits._

/* InputActivationSubsystem
 * Handles the input data stream, and stores it in to a input buffer. It also "rolls" through the input activation
 * as a convolution opperation would; and does so continously until the next signal is asserted. This allows looping
 * through the input to convolve it with more than one kernel.
 */
class InputActivationsSubsystem(input: lbir.QTensor, kernel: lbir.QTensor, output: lbir.QTensor, options: LayerOptions)
    extends Module {
  val io = IO(new Bundle {
    val inStream = Flipped(AXIStream(UInt(options.busWidthIn.W)))
    val inputActivationsWindow = Decoupled(Vec(kernel.numKernelParams, UInt(input.dtype.bitwidth.W)))
    val channelDone = Output(Bool())
  })
  val actMem = Module(MemoryGenerator.SRAM(depth = input.memDepth, width = MemWordSize.bits))
  val dataMover = Module(new InputDataMover(input))
  val shiftRegConvolver = Module(new ShiftRegisterConvolver(input, kernel, output))

  object InSubState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sRECEVING_DATA = Value(1.U)
    val sFULL = Value(2.U)
  }
  val state = RegInit(InSubState.sEMPTY)

  /* INPUT STREAM LOGIC*/
  val inputTensorFinnished = state === InSubState.sEMPTY && RegNext(state === InSubState.sFULL)
  val (actMemCntValue, _) = Counter(0 to input.memDepth, io.inStream.fire, inputTensorFinnished)
  io.inStream.ready := state =/= InSubState.sFULL
  actMem.io.write.address := actMemCntValue
  actMem.io.write.data := io.inStream.bits
  actMem.io.write.enable := io.inStream.fire

  dataMover.io.actMem <> actMem.io.read
  dataMover.io.actMemWrittenTo := actMemCntValue

  val startOfTransmission = state === InSubState.sRECEVING_DATA && RegNext(state === InSubState.sEMPTY)
  dataMover.io.start := RegNext(startOfTransmission) // Start one cycle after start of transmission of the input packet

  shiftRegConvolver.io.nextElement <> dataMover.io.nextElement
  io.inputActivationsWindow <> shiftRegConvolver.io.inputActivationsWindow
  io.channelDone := shiftRegConvolver.io.channelDone

  when(state === InSubState.sEMPTY && io.inStream.fire) {
    state := InSubState.sRECEVING_DATA
  }.elsewhen(state === InSubState.sRECEVING_DATA && actMemCntValue === (input.memDepth - 1).U) {
    state := InSubState.sFULL
  }.otherwise {
    when(dataMover.io.done) {
      state := InSubState.sEMPTY
    }
  }
}
