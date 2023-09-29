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

import chisel3._
import chisel3.util._
import chisel4ml.implicits._

/** A register file for storing the inputs (activations or image) of a convotional layer.
  */
class RollingRegisterFile(input: lbir.QTensor, kernel: lbir.QTensor) extends Module {
  val io = IO(new Bundle {
    val shiftRegs = Input(Bool())
    val rowWriteMode = Input(Bool())
    val rowAddr = Input(UInt(log2Up(kernel.width).W))
    val chAddr = Input(UInt(log2Up(kernel.numChannels).W))
    val inData = Input(UInt((kernel.width * input.dtype.bitwidth).W))
    val inValid = Input(Bool())
    val outData = Output(UInt((kernel.numKernelParams * input.dtype.bitwidth).W))
  })

  val regs = RegInit(VecInit.fill(kernel.numChannels, kernel.width, kernel.height)(0.U(input.dtype.bitwidth.W)))
  io.outData := regs.asUInt

  regs := regs
  when(io.inValid) {
    when(io.rowWriteMode === true.B) {
      regs(io.chAddr)(io.rowAddr) := io.inData.asTypeOf(Vec(kernel.width, UInt(input.dtype.bitwidth.W)))
    }.otherwise {
      for (i <- 0 until kernel.width) {
        regs(io.chAddr)(i)(kernel.width - 1) := io.inData.asTypeOf(Vec(kernel.width, UInt(input.dtype.bitwidth.W)))(i)
      }
    }
  }

  when(io.shiftRegs === true.B) {
    for {
      i <- 0 until kernel.numChannels
      k <- 0 until kernel.width - 1
      j <- 0 until kernel.width
    } {
      regs(i)(j)(k) := regs(i)(j)(k + 1)
    }
  }
}
