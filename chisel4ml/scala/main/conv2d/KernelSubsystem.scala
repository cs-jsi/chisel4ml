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
import chisel4ml.conv2d.KernelRegisterFile

class KernelThreshIO[I <: Bits, A <: Bits](qt: lbir.QTensor, genThresh: A) extends Bundle {
  val kernel = Output(UInt((qt.numKernelParams * qt.dtype.bitwidth).W))
  val thresh = new ThreshAndShiftIO(genThresh)
}

class KernelControlIO(numberOfKernels: Int) extends Bundle {
  val ready = Output(Bool())
  val loadKernel = Input(Valid(UInt(log2Up(numberOfKernels).W)))
}

class KernelSubsystem[I <: Bits, A <: Bits](kernelQt: lbir.QTensor, thresh: lbir.QTensor, genThresh: A) extends Module {
  val io = IO(new Bundle {
    val kernel = Valid(new KernelThreshIO(kernelQt, genThresh))
    val ctrl = new KernelControlIO(kernelQt.numKernels)
  })

  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = kernelQt.toHexStr, width = MemWordSize.bits))
  val kRFLoader = Module(new KernelRFLoader(kernelQt))
  val krf = Module(new KernelRegisterFile(kernelQt))
  val tas = Module(new ThreshAndShiftUnit[A](genThresh, thresh, kernelQt))

  kernelMem.io.write.enable := false.B // io.kernelMemWrEna
  kernelMem.io.write.address := 0.U // io.kernelMemWrAddr
  kernelMem.io.write.data := 0.U // io.kernelMemWrData
  kernelMem.io.read <> kRFLoader.io.rom

  krf.io.write <> kRFLoader.io.krf

  io.ctrl <> kRFLoader.io.ctrl

  io.kernel.bits.thresh <> tas.tasIO
  io.kernel.bits.kernel := krf.io.kernel
  io.kernel.valid := false.B

  tas.loadKernel := io.ctrl.loadKernel
}
