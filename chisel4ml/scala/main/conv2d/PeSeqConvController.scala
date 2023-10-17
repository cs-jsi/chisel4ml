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

object ctrlState extends ChiselEnum {
  val sWAITFORDATA = Value(0.U)
  val sLOADINPACT = Value(1.U)
  val sLOADKERNEL = Value(2.U)
  val sCOMP = Value(3.U)
  val sWAITWRITE = Value(4.U)
  val sWAITWRITE2 = Value(5.U)
  val sSENDDATA = Value(6.U)
}

/** PeSeqConvController
  */
class PeSeqConvController(layer: Conv2DConfig) extends Module {
  val io = IO(new Bundle {
    val loadKernel = Output(Valid(UInt(log2Up(layer.kernel.numKernels).W)))
    val nextInputTensor = Output(Bool())
  })
  io.loadKernel.valid := false.B
  io.loadKernel.bits := 0.U
  io.nextInputTensor := false.B
}
