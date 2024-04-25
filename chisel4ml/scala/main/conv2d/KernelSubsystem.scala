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

class ThreshAndShiftIO[A <: Bits](threshold: lbir.QTensor) extends Bundle {
  val thresh = threshold.getType[A]
  val shift = UInt(8.W)
  val shiftLeft = Bool()
}

class KernelSubsystemIO[W <: Bits, A <: Bits](kernel: lbir.QTensor, threshold: lbir.QTensor, depthwise: Boolean)
    extends Bundle {
  val activeKernel = Vec(kernel.numActiveParams(depthwise), kernel.getType[W])
  val threshShift = new ThreshAndShiftIO[A](threshold)
}

class KernelRegisterFile[W <: Bits](kernel: lbir.QTensor, depthwise: Boolean) extends Module {
  val io = IO(new Bundle {
    val write = Flipped(Valid(kernel.getType[W]))
    val activeKernel = Valid(Vec(kernel.numActiveParams(depthwise), kernel.getType[W]))
  })
  val valid = RegInit(false.B)
  val regs = RegInit(VecInit(Seq.fill(kernel.numActiveParams(depthwise))(0.U.asTypeOf(kernel.getType[W]))))
  val (regCnt, regCntWrap) = Counter(0 until kernel.numActiveParams(depthwise), io.write.valid)
  when(regCntWrap) {
    valid := true.B
  }.elsewhen(valid && io.write.fire) {
    valid := false.B
  }

  when(io.write.valid) {
    regs(regCnt) := io.write.bits
  }

  io.activeKernel.bits := regs
  io.activeKernel.valid := valid
}

class KernelSubsystem[W <: Bits, A <: Bits](l: lbir.Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    val weights = Valid(new KernelSubsystemIO[W, A](l.kernel, l.thresh, l.depthwise))
    val ctrl = new KernelRFLoaderControlIO(l)
  })
  val kernelMem = Module(
    MemoryGenerator.SRAMInitFromString(hexStr = l.kernel.toHexString(), width = 32, noWritePort = true)
  )
  val kRFLoader = Module(new KernelRFLoader[W](l))
  val krf = Module(new KernelRegisterFile[W](l.kernel, l.depthwise))
  val tasu = Module(new ThreshAndShiftUnit[A](l.thresh, l.kernel))

  kRFLoader.io.rom <> kernelMem.io.read
  krf.io.write <> kRFLoader.io.krf

  io.weights.bits.activeKernel := krf.io.activeKernel.bits
  io.weights.bits.threshShift <> tasu.io.tas
  io.weights.valid := krf.io.activeKernel.valid

  kRFLoader.io.ctrl <> io.ctrl
  tasu.io.loadKernel <> io.ctrl.loadKernel
}
