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

import memories.MemoryGenerator
import _root_.chisel4ml.sequential._
import _root_.chisel4ml.util._
import _root_.chisel4ml.implicits._
import _root_.lbir.QTensor
import chisel3._
import chisel3.util._

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
    val kernelReady = Output(Bool())
    val loadKernel = Input(Bool())
    val kernelNum = Input(UInt(log2Up(numKernels).W))

    val krfOutput = Output(UInt(outDataSize.W))
  })

  val krfLoader = Module(new KernelRFLoader(kernel = parameters))

  val krf = Module(new KernelRegisterFile(kernel = parameters))

  val kernelMem = Module(MemoryGenerator.SRAMInitFromString(hexStr = parameters.toHexStr, width = memWordWidth))

  krf.io.chAddr := krfLoader.io.chAddr
  krf.io.rowAddr := krfLoader.io.rowAddr
  krf.io.colAddr := krfLoader.io.colAddr
  krf.io.inData := krfLoader.io.data
  krf.io.inValid := krfLoader.io.valid

  kernelMem.io.rdEna := krfLoader.io.romRdEna
  kernelMem.io.rdAddr := krfLoader.io.romRdAddr
  krfLoader.io.romRdData := kernelMem.io.rdData
  kernelMem.io.wrData := 0.U
  kernelMem.io.wrEna := false.B
  kernelMem.io.wrAddr := 0.U

  krfLoader.io.loadKernel := io.loadKernel
  krfLoader.io.kernelNum := io.kernelNum
  io.kernelReady := krfLoader.io.kernelReady

  io.krfOutput := krf.io.outData
}
