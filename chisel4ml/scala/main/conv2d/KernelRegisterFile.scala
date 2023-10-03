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

class KernelRegisterFileInput(qt: lbir.QTensor) extends Bundle {
  val channelAddress = UInt(log2Up(qt.numChannels).W)
  val rowAddress = UInt(log2Up(qt.width).W)
  val columnAddress = UInt(log2Up(qt.height).W)
  val data = UInt(qt.dtype.bitwidth.W)
}

class KernelRegisterFileIO(qt: lbir.QTensor) extends Bundle {
  val write = Input(Valid(new KernelRegisterFileInput(qt)))
  val kernel = UInt((qt.numKernelParams * qt.dtype.bitwidth).W)
}

/** A register file for storing the weights/kernel of a convolution layer.
  *
  * kernelSize: Int - Signifies one dimension of a square kernel (If its a 3x3 kernel then kernelSize=3) kernelDepth:
  * kernelDepth: Int - The depth of the kernel (number of input channels) actParamSize: Int - The activation
  * parameterSize in bits. kernelParamSize: Int - Bitwidth of each kernel parameter.
  */
class KernelRegisterFile(kernel: lbir.QTensor) extends Module {
  val io = IO(new KernelRegisterFileIO(kernel))

  val regs = RegInit(VecInit.fill(kernel.numChannels, kernel.width, kernel.height)(0.U(kernel.dtype.bitwidth.W)))

  when(io.write.valid) {
    regs(io.write.bits.channelAddress)(io.write.bits.rowAddress)(io.write.bits.columnAddress) := io.write.bits.data
  }

  io.kernel := regs.asUInt
}
