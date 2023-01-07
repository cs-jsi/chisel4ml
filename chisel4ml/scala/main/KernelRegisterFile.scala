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
package chisel4ml.sequential

import _root_.chisel4ml.util.reqWidth
import chisel3._

/** A register file for storing the weights/kernel of a convolution layer.
  *
  * kernelSize: Int - Signifies one dimension of a square kernel (If its a 3x3 kernel then kernelSize=3) kernelDepth:
  * kernelDepth: Int - The depth of the kernel (number of input channels) actParamSize: Int - The activation
  * parameterSize in bits. kernelParamSize: Int - Bitwidth of each kernel parameter.
  */
class KernelRegisterFile(kernelSize: Int, kernelDepth: Int, kernelParamSize: Int) extends Module {
  val totalNumOfElements:  Int = kernelSize * kernelSize * kernelDepth
  val kernelNumOfElements: Int = kernelSize * kernelSize
  val wrDataWidth:         Int = kernelNumOfElements * kernelParamSize
  val outDataSize:         Int = kernelSize * kernelSize * kernelDepth * kernelParamSize
  val kernelAddrWidth:     Int = reqWidth(kernelDepth)
  val rowAddrWidth:        Int = reqWidth(kernelSize)

  val io = IO(new Bundle {
    val kernelAddr = Input(UInt(kernelAddrWidth.W))
    val inData     = Input(UInt(wrDataWidth.W))
    val inValid    = Input(Bool())
    val outData    = Output(UInt(outDataSize.W))
  })

  val regs = RegInit(VecInit.fill(kernelDepth, kernelNumOfElements)(0.U(kernelParamSize.W)))

  when(io.inValid) {
    regs(io.kernelAddr) := io.inData.asTypeOf(Vec(kernelNumOfElements, UInt(kernelParamSize.W)))
  }

  io.outData := regs.asUInt
}
