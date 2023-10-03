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
package chisel4ml.tests

import _root_.chisel4ml.implicits._
import _root_.chisel4ml.conv2d._
import _root_.lbir.QTensor
import chisel3._
import memories.MemoryGenerator

/** Kernel Register File Loader - test bed
  */

class KernelRFLoaderTestBed(
  kernelSize:      Int,
  kernelDepth:     Int,
  kernelParamSize: Int,
  numKernels:      Int,
  parameters:      QTensor)
    extends Module {

  val memWordWidth:                Int = 32
  val kernelParamsPerWord:         Int = memWordWidth / kernelParamSize
  val totalNumOfElementsPerKernel: Int = kernelSize * kernelSize * kernelDepth
  val outDataSize:                 Int = totalNumOfElementsPerKernel * kernelParamSize
  val wordsPerKernel:              Int = math.ceil(totalNumOfElementsPerKernel.toFloat / kernelParamsPerWord.toFloat).toInt
  val kernelMemDepthWords:         Int = wordsPerKernel * numKernels

  val io = IO(new Bundle {
    val ctrl = new KernelControlIO(parameters.numKernels)
    val krfOutput = Output(UInt(outDataSize.W))
  })

  val krfLoader = Module(new KernelRFLoader(kernel = parameters))

  val krf = Module(new KernelRegisterFile(kernel = parameters))

  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = parameters.toHexStr, width = memWordWidth))

  krf.io.write <> krfLoader.io.krf

  kernelMem.io.read.enable := krfLoader.io.rom.enable
  kernelMem.io.read.address := krfLoader.io.rom.address
  krfLoader.io.rom.data := kernelMem.io.read.data
  kernelMem.io.write.data := 0.U
  kernelMem.io.write.enable := false.B
  kernelMem.io.write.address := 0.U

  krfLoader.io.ctrl <> io.ctrl

  io.krfOutput := krf.io.kernel
}
