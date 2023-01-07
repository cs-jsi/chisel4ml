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

/** A register file for storing the inputs (activations or iamge) of a convotional layer.
  *
  * kernelSize: Int - Signifies one dimension of a square kernel (If its a 3x3 kernel then kernelSize=3) kernelDepth:
  * Int - The depth of the kernel (number of input channels) actParamSize: Int - The activation parameterSize in bits.
  */
class RollingRegisterFile(kernelSize: Int, kernelDepth: Int, paramSize: Int) extends Module {
  val totalNumOfElements:  Int = kernelSize * kernelSize * kernelDepth
  val kernelNumOfElements: Int = kernelSize * kernelSize
  val outDataSize:         Int = kernelSize * kernelSize * kernelDepth * paramSize
  val wrDataWidth:         Int = kernelSize * paramSize
  val chAddrWidth:         Int = reqWidth(kernelDepth)
  val rowAddrWidth:        Int = reqWidth(kernelSize)

  val io = IO(new Bundle {
    val shiftRegs    = Input(Bool())
    val rowWriteMode = Input(Bool())
    val rowAddr      = Input(UInt(rowAddrWidth.W))
    val chAddr       = Input(UInt(chAddrWidth.W))
    val inData       = Input(UInt(wrDataWidth.W))
    val inValid      = Input(Bool())
    val outData      = Output(UInt(outDataSize.W))
  })

  val regs = RegInit(VecInit.fill(kernelDepth, kernelSize, kernelSize)(0.U(paramSize.W)))
  io.outData := regs.asUInt

  when(io.inValid) {
    when(io.rowWriteMode === true.B) {
      regs(io.chAddr)(io.rowAddr) := io.inData.asTypeOf(Vec(kernelSize, UInt(paramSize.W)))
    }.otherwise {
      for (i <- 0 until kernelSize) {
        regs(io.chAddr)(i)(kernelSize - 1) := io.inData.asTypeOf(Vec(kernelSize, UInt(paramSize.W)))(
          i,
        )
      }
      when(io.shiftRegs === true.B) {
        for {
          i <- 0 until kernelDepth
          k <- 0 until kernelSize - 1
          j <- 0 until kernelSize
        } {
          regs(i)(j)(k) := regs(i)(j)(k + 1)
        }
      }
    }
  }
}
