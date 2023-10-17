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
class InputActivationsSubsystem(input: lbir.QTensor, kernel: lbir.QTensor, options: LayerOptions) extends Module {
  val io = IO(new Bundle {
    val inStream = Flipped(AXIStream(UInt(options.busWidthIn.W)))
    val data = Output(Decoupled(UInt((kernel.numKernelParams * input.dtype.bitwidth).W)))
    val next = Input(Bool())
  })
  val actMem = Module(MemoryGenerator.SRAM(depth = input.memDepth, width = MemWordSize.bits))
  val dataMover = Module(new InputDataMover(input, kernel))
  val shiftRegConvolver = Module(new ShiftRegisterConvolver(input, kernel))

  object InSubState extends ChiselEnum {
    val sEMPTY = Value(0.U)
    val sRECEVING_DATA = Value(1.U)
    val sFULL = Value(2.U)
  }
  val state = RegInit(ctrlState.sWAITFORDATA)
  val nstate = WireInit(ctrlState.sCOMP)

  /* INPUT STREAM LOGIC*/
  /*val (actMemCntValue, actMemCntWrap) = Counter(io.inStream.fire, input.memDepth)
  io.inStream.ready := state =/= InSubState.sFULL


  actMem.io.read <> swu.io.actMem

  actRegFile.io.shiftRegs := swu.io.shiftRegs
  actRegFile.io.rowWriteMode := swu.io.rowWriteMode
  actRegFile.io.rowAddr := swu.io.rowAddr
  actRegFile.io.chAddr := swu.io.chAddr
  actRegFile.io.inData := swu.io.data
  actRegFile.io.inValid := swu.io.valid

  io.data.bits := actRegFile.io.outData
  io.data.valid := RegNext(swu.io.imageValid)

  swu.io.start := io.start
  io.end := swu.io.end

  io.inStream.ready :=  io.state === ctrlState.sLOADINPACT || io.state === ctrlState.sWAITFORDATA
  actMem.io.write.enable := io.inStream.ready && io.inStream.valid
  actMem.io.write.address := actMemCnt
  actMem.io.write.data := io.inStream.bits*/
}
