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
  val numVirtualChannels = if (l.depthwise) l.kernel.numChannels else 1
  val (_, virtualCntWrap) = Counter(0 until numVirtualChannels, io.activeDone)
  val (kernelCntVal, _) = Counter(0 until l.kernel.numKernels, io.kernelCtrl.kernelDone)

  io.kernelCtrl.nextActive.foreach(_ := RegNext(io.activeDone))
  io.kernelCtrl.loadKernel.bits := kernelCntVal
  io.kernelCtrl.loadKernel.valid := RegNext(virtualCntWrap || reset.asBool)
}
