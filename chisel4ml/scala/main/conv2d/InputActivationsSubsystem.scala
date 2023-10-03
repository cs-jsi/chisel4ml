package chisel4ml.conv2d

import chisel3._
import chisel3.util._
import interfaces.amba.axis.AXIStream
import services.LayerOptions
import memories.MemoryGenerator
import chisel4ml.MemWordSize
import chisel4ml.implicits._

class InputActivationsSubsystem(input: lbir.QTensor, kernel: lbir.QTensor, options: LayerOptions) extends Module {
  val io = IO(new Bundle {
    val inStream = Flipped(AXIStream(UInt(options.busWidthIn.W)))
    val outData = Output(Valid(UInt((kernel.numKernelParams * input.dtype.bitwidth).W)))
    val start = Input(Bool())
    val end = Output(Bool())
    val inStreamReady = Input(Bool())
    val inStreamValid = Output(Bool())
    val inStreamLast = Output(Bool())
    val actMemAddr = Input(UInt(input.memDepth.W))
  })
  val actMem = Module(MemoryGenerator.SRAM(depth = input.memDepth, width = MemWordSize.bits))
  val swu = Module(new SlidingWindowUnit(input = input, kernel = kernel))
  val actRegFile = Module(new RollingRegisterFile(input, kernel))

  actMem.io.read <> swu.io.actMem

  actRegFile.io.shiftRegs := swu.io.shiftRegs
  actRegFile.io.rowWriteMode := swu.io.rowWriteMode
  actRegFile.io.rowAddr := swu.io.rowAddr
  actRegFile.io.chAddr := swu.io.chAddr
  actRegFile.io.inData := swu.io.data
  actRegFile.io.inValid := swu.io.valid

  io.outData.bits := actRegFile.io.outData
  io.outData.valid := RegNext(swu.io.imageValid)

  swu.io.start := io.start
  io.end := swu.io.end

  io.inStream.ready := io.inStreamReady
  actMem.io.write.enable := io.inStream.ready && io.inStream.valid
  actMem.io.write.address := io.actMemAddr
  actMem.io.write.data := io.inStream.bits
  io.inStreamLast := io.inStream.last
  io.inStreamValid := io.inStream.valid
}
