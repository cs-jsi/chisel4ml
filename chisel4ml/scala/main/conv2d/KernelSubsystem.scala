/*
 * Copyright 2022 Computer Systems Department, Jozef Stefan Insitute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package chisel4ml.conv2d
import chisel3._
import chisel3.util._
import chisel4ml.implicits._
import memories.MemoryGenerator
import chisel4ml.MemWordSize

class KernelRegisterFileInput(kernel: lbir.QTensor) extends Bundle {
  val channelAddress = UInt(log2Up(kernel.numChannels).W)
  val rowAddress = UInt(log2Up(kernel.width).W)
  val columnAddress = UInt(log2Up(kernel.height).W)
  val data = UInt(kernel.dtype.bitwidth.W)
}

class ThreshAndShiftIO[A <: Bits](genThresh: A) extends Bundle {
  val thresh = genThresh.cloneType
  val shift = UInt(8.W)
  val shiftLeft = Bool()
}

class KernelSubsystemIO[A <: Bits](kernel: lbir.QTensor, genThresh: A) extends Bundle {
  val activeKernel = UInt((kernel.numKernelParams * kernel.dtype.bitwidth).W)
  val threshShift = new ThreshAndShiftIO(genThresh)
}

class KernelRegisterFile(kernel: lbir.QTensor) extends Module {
  val io = IO(new Bundle {
    val write = Flipped(Valid(new KernelRegisterFileInput(kernel)))
    val activeKernel = UInt((kernel.numKernelParams * kernel.dtype.bitwidth).W)
  })
  val regs = RegInit(VecInit.fill(kernel.numChannels, kernel.width, kernel.height)(0.U(kernel.dtype.bitwidth.W)))

  when(io.write.valid) {
    regs(io.write.bits.channelAddress)(io.write.bits.rowAddress)(io.write.bits.columnAddress) := io.write.bits.data
  }

  io.activeKernel := regs.asUInt
}

class KernelSubsystem[I <: Bits, A <: Bits](kernel: lbir.QTensor, thresh: lbir.QTensor, genThresh: A) extends Module {
  val io = IO(new Bundle {
    val weights = Valid(new KernelSubsystemIO(kernel, genThresh))
    val loadKernel = Flipped(Valid(UInt(log2Up(kernel.numKernels).W)))
  })
  object KernelSubState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sLOADING = Value(1.U)
    val sREADY = Value(2.U)
  }
  val state = RegInit(KernelSubState.sWAIT)

  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = kernel.toHexStr, width = MemWordSize.bits))
  val kRFLoader = Module(new KernelRFLoader(kernel))
  val krf = Module(new KernelRegisterFile(kernel))
  val tasu = Module(new ThreshAndShiftUnit[A](genThresh, thresh, kernel))

  kernelMem.io.write <> MemoryGenerator.getTieOffBundle(depth = kernel.memDepth, width = MemWordSize.bits)
  kRFLoader.io.rom <> kernelMem.io.read
  krf.io.write <> kRFLoader.io.krf

  io.weights.bits.activeKernel := krf.io.activeKernel
  io.weights.bits.threshShift <> tasu.io.tas
  io.weights.valid := state === KernelSubState.sREADY

  kRFLoader.io.loadKernel <> io.loadKernel
  tasu.io.loadKernel <> io.loadKernel

  when(state =/= KernelSubState.sLOADING && io.loadKernel.valid) {
    state := KernelSubState.sLOADING
  }.elsewhen(state === KernelSubState.sLOADING && kRFLoader.io.kernelLoaded) {
    state := KernelSubState.sREADY
  }
}
