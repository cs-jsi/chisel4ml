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
    val inputActivationsWindow = Decoupled(Vec(l.kernel.numActiveParams(true), l.input.getType[I]))
    val channelDone = Output(Bool())
  })

  object SRCState extends ChiselEnum {
    val sNOT_FULL = Value(0.U)
    val sFULL = Value(1.U)
    val sSTALL = Value(2.U)
  }
  val state = RegInit(SRCState.sNOT_FULL)

  val numRegs: Int = l.input.width * l.kernel.height - (l.input.width - l.kernel.width)
  val regs = ShiftRegisters(io.nextElement.bits, numRegs, io.nextElement.fire)

  val (elementCounter, _) = Counter(0 to l.input.numActiveParams(true), io.nextElement.fire, io.channelDone)
  val (inputWidthCounter, inputWidthCounterWrap) = Counter(0 until l.input.width, io.nextElement.fire, io.channelDone)
  val (outputCounter, outputCounterWrap) =
    Counter(0 until l.output.numActiveParams(true), io.inputActivationsWindow.fire)
  dontTouch(outputCounter)
  dontTouch(elementCounter)

  io.channelDone := outputCounterWrap
  io.inputActivationsWindow.bits.zipWithIndex.foreach {
    case (elem, ind) => elem := regs(transformIndex(ind))
  }

  io.inputActivationsWindow.valid := state === SRCState.sFULL && (inputWidthCounter > (l.kernel.width - 1).U || RegNext(
    inputWidthCounterWrap
  ))
  io.nextElement.ready := (state === SRCState.sNOT_FULL || (state =/= SRCState.sNOT_FULL && io.inputActivationsWindow.ready)) && !io.channelDone

  when(io.channelDone) {
    assert(state === SRCState.sFULL)
    state := SRCState.sNOT_FULL
  }.elsewhen(state === SRCState.sNOT_FULL && elementCounter > (numRegs - 2).U && io.nextElement.fire) {
    state := SRCState.sFULL
  }.elsewhen(state === SRCState.sFULL && !io.nextElement.fire && io.inputActivationsWindow.fire) {
    state := SRCState.sSTALL
  }.elsewhen(state === SRCState.sSTALL && io.nextElement.fire) {
    assert(io.inputActivationsWindow.valid === false.B)
    state := SRCState.sFULL
  }
}
