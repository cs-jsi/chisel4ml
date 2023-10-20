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

/** PeSeqConvController
  */
class PeSeqConvController(layer: Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    val loadKernel = Output(Valid(UInt(log2Up(layer.kernel.numKernels).W)))
    val channelDone = Input(Bool())
  })
  val (chCntVal, _) = Counter(0 until layer.kernel.numChannels, io.channelDone)
  object CtrlState extends ChiselEnum {
    val sLOADKERNEL = Value(0.U)
    val sCOMP = Value(1.U)
  }
  val state = RegInit(CtrlState.sLOADKERNEL)
  when(state === CtrlState.sLOADKERNEL) {
    state := CtrlState.sCOMP
  }.elsewhen(state === CtrlState.sCOMP && io.channelDone) {
    state := CtrlState.sLOADKERNEL
  }
  io.loadKernel.valid := state === CtrlState.sLOADKERNEL
  io.loadKernel.bits := chCntVal
}
