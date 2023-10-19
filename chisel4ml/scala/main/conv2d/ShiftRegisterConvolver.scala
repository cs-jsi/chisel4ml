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

/* ShiftRegisterConvolver
 *
 * Shifts the inputs in standard LBIR order (one by one), and generates the active output window.
 *
 */
class ShiftRegisterConvolver(input: lbir.QTensor, kernel: lbir.QTensor, output: lbir.QTensor) extends Module {
  require(kernel.numChannels == 1, "Module only works with single channel input Conv2D or DepthWise Conv2D.")
  private def transformIndex(ind: Int): Int = {
    require(ind >= 0 && ind < kernel.numKernelParams)
    val revInd = (kernel.numKernelParams - 1) - ind // The first element is in the last position
    input.width * (revInd / kernel.width) + (revInd % kernel.width)
  }

  val io = IO(new Bundle {
    val nextElement = Flipped(Decoupled(UInt(input.dtype.bitwidth.W)))
    val inputActivationsWindow = Decoupled(Vec(kernel.numKernelParams, UInt(input.dtype.bitwidth.W)))
    val channelDone = Output(Bool())
  })

  val numRegs = input.width * kernel.height - (input.width - kernel.width)
  val regs = ShiftRegisters(io.nextElement.bits, numRegs, io.nextElement.fire)
  val (regsFilledCntValue, _) =
    Counter(0 until (input.numKernelParams + kernel.height - 1), io.nextElement.fire, io.channelDone)
  val (lineCntValue, _) =
    Counter(0 until input.width, regsFilledCntValue >= numRegs.U && io.nextElement.fire, io.channelDone)
  val (_, outputCntWrap) = Counter(0 until output.numKernelParams, io.inputActivationsWindow.fire)

  io.channelDone := outputCntWrap
  io.inputActivationsWindow.bits.zipWithIndex.foreach {
    case (elem: UInt, ind: Int) => elem := regs(transformIndex(ind))
  }

  io.nextElement.ready := io.inputActivationsWindow.ready && !io.channelDone
  // First condition: waiting for shift registers to initially fill up. Second: filter goes over the width of image. Third: backpressure from input data mover.
  io.inputActivationsWindow.valid := (regsFilledCntValue >= numRegs.U) && (lineCntValue <= (input.width - kernel.width).U) && RegNext(
    io.nextElement.fire
  )
}
