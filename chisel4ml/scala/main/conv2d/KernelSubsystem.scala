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

class ThreshAndShiftIO[A <: Bits](genThresh: A) extends Bundle {
  val thresh = genThresh.cloneType
  val shift = UInt(8.W)
  val shiftLeft = Bool()
}

class KernelSubsystemIO[A <: Bits](kernel: lbir.QTensor, genThresh: A) extends Bundle {
  val activeKernel = UInt((kernel.numKernelParams * kernel.dtype.bitwidth).W)
  val threshShift = new ThreshAndShiftIO(genThresh)
}

class KernelRegisterFile(kernel: lbir.QTensor, depthwise: Boolean) extends Module {
  val io = IO(new Bundle {
    val write = Flipped(Valid(UInt(kernel.dtype.bitwidth.W)))
    val activeKernel = Valid(UInt((kernel.numActiveParams(depthwise) * kernel.dtype.bitwidth).W))
  })
  val valid = RegInit(false.B)
  val numVirtualChannels = if (depthwise) 1 else kernel.numChannels
  val regs = RegInit(VecInit.fill(numVirtualChannels * kernel.width * kernel.height)(0.U(kernel.dtype.bitwidth.W)))
  val (regCnt, regCntWrap) = Counter(0 until (numVirtualChannels * kernel.width * kernel.height), io.write.valid)
  when(regCntWrap) {
    valid := true.B
  }.elsewhen(valid && io.write.fire) {
    valid := false.B
  }

  when(io.write.valid) {
    regs(regCnt) := io.write.bits
  }

  io.activeKernel.bits := regs.asUInt
  io.activeKernel.valid := valid
}

class KernelSubsystem[I <: Bits, A <: Bits](l: lbir.Conv2DConfig, genThresh: A) extends Module {
  val io = IO(new Bundle {
    val weights = Valid(new KernelSubsystemIO(l.kernel, genThresh))
    val ctrl = new KernelRFLoaderControlIO(l)
  })
  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = l.kernel.toHexStr, width = MemWordSize.bits))
  val kRFLoader = Module(new KernelRFLoader(l))
  val krf = Module(new KernelRegisterFile(l.kernel, l.depthwise))
  val tasu = Module(new ThreshAndShiftUnit[A](genThresh, l.thresh, l.kernel))

  kernelMem.io.write <> MemoryGenerator.getTieOffBundle(depth = l.kernel.memDepth, width = MemWordSize.bits)
  kRFLoader.io.rom <> kernelMem.io.read
  krf.io.write <> kRFLoader.io.krf

  io.weights.bits.activeKernel := krf.io.activeKernel.bits
  io.weights.bits.threshShift <> tasu.io.tas
  io.weights.valid := krf.io.activeKernel.valid

  kRFLoader.io.ctrl <> io.ctrl
  tasu.io.loadKernel <> io.ctrl.loadKernel
}
