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
  val outDataSize: Int = kernelSize * kernelSize * kernelDepth * kernelParamSize

  val io = IO(new Bundle {
    val chAddr     = Input(UInt(reqWidth(kernelDepth).W))
    val rowAddr    = Input(UInt(reqWidth(kernelSize).W))
    val colAddr    = Input(UInt(reqWidth(kernelSize).W))
    val inData     = Input(UInt(kernelParamSize.W))
    val inValid    = Input(Bool())
    val outData    = Output(UInt(outDataSize.W))
  })

  val regs = RegInit(VecInit.fill(kernelDepth, kernelSize, kernelSize)(0.U(kernelParamSize.W)))

  when(io.inValid) {
    regs(io.chAddr)(io.rowAddr)(io.colAddr) := io.inData
  }

  io.outData := regs.asUInt
}
