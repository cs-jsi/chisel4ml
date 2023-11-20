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
class PeSeqConvController(l: Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    val kernelCtrl = Flipped(new KernelRFLoaderControlIO(l))
    val activeDone = Input(Bool())
  })

  object PeSeqConvState extends ChiselEnum {
    val sLOAD_KERNEL = Value(0.U)
    val sCOMPUTING = Value(1.U)
    val sLAST_ACTIVE = Value(2.U)
  }
  val state = RegInit(PeSeqConvState.sLOAD_KERNEL)

  val numVirtualChannels = if (l.depthwise) l.kernel.numChannels else 1
  val (virtualChannelsCounter, virtualChannelsCounterWrap) =
    Counter(0 until numVirtualChannels, io.activeDone, state === PeSeqConvState.sLOAD_KERNEL)
  val (kernelCounter, _) = Counter(0 until l.kernel.numKernels, io.kernelCtrl.lastActiveLoaded)

  ///////////////////////
  // NEXT STATE LOGIC  //
  ///////////////////////
  when(state === PeSeqConvState.sLOAD_KERNEL) {
    assert(virtualChannelsCounter === 0.U)
    state := PeSeqConvState.sCOMPUTING
  }.elsewhen(state === PeSeqConvState.sCOMPUTING && io.kernelCtrl.lastActiveLoaded) {
    state := PeSeqConvState.sLAST_ACTIVE
  }.elsewhen((state === PeSeqConvState.sLAST_ACTIVE) && io.activeDone) {
    state := PeSeqConvState.sLOAD_KERNEL
  }

  io.kernelCtrl.nextActive.foreach(_ := RegNext(io.activeDone))
  io.kernelCtrl.loadKernel.bits := kernelCounter
  io.kernelCtrl.loadKernel.valid := state === PeSeqConvState.sLOAD_KERNEL
}
