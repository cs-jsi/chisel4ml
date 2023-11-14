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
import lbir.Conv2DConfig

/* ShiftRegisterConvolver
 *
 * Shifts the inputs in standard LBIR order (one by one), and generates the active output window.
 *
 */
class ShiftRegisterConvolver[I <: Bits](l: Conv2DConfig) extends Module {
  require(
    (l.kernel.numChannels == 1 && l.depthwise == false) || l.depthwise == true,
    s"""Module only works with single channel input Conv2D or DepthWise Conv2D.
       | Instead the kernel has shape ${l.kernel.shape}.""".stripMargin.replaceAll("\n", "")
  )
  private def transformIndex(ind: Int): Int = {
    require(ind >= 0 && ind < l.kernel.numActiveParams(l.depthwise))
    val revInd = (l.kernel.numActiveParams(l.depthwise) - 1) - ind // The first element is in the last position
    l.input.width * (revInd / l.kernel.width) + (revInd % l.kernel.width)
  }

  val io = IO(new Bundle {
    val nextElement = Flipped(Decoupled(l.input.getType[I]))
    val inputActivationsWindow = Decoupled(Vec(l.kernel.numActiveParams(l.depthwise), l.input.getType[I]))
    val channelDone = Output(Bool())
  })

  val numRegs = l.input.width * l.kernel.height - (l.input.width - l.kernel.width)
  val regs = ShiftRegisters(io.nextElement.bits, numRegs, io.nextElement.fire)
  val fireCntMax = (l.input.numActiveParams(true) + l.kernel.height - 1)
  val (fireCntVal, _) = Counter(0 until fireCntMax, io.nextElement.fire, io.channelDone)
  val (lineCntValue, _) = Counter(0 until l.input.width, fireCntVal >= numRegs.U && io.nextElement.fire, io.channelDone)
  val (_, outputCntWrap) = Counter(0 until l.output.numActiveParams(true), io.inputActivationsWindow.fire)

  io.channelDone := outputCntWrap
  io.inputActivationsWindow.bits.zipWithIndex.foreach {
    case (elem, ind) => elem := regs(transformIndex(ind))
  }

  io.nextElement.ready := io.inputActivationsWindow.ready && !io.channelDone
  // First condition: waiting for shift registers to initially fill up. Second: filter goes over the width of image. Third: backpressure from input data mover.
  io.inputActivationsWindow.valid := (fireCntVal >= numRegs.U) && (lineCntValue <= (l.input.width - l.kernel.width).U) && RegNext(
    io.nextElement.fire
  )
}
