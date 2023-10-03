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

class ThreshAndShiftIO[A <: Bits](genThresh: A) extends Bundle {
  val thresh = Output(genThresh)
  val shift = Output(UInt(8.W))
  val shiftLeft = Output(Bool())
}

/** ThreshAndShiftUnit
  */
class ThreshAndShiftUnit[A <: Bits](genThresh: A, thresh: lbir.QTensor, kernel: lbir.QTensor) extends Module {
  val tasIO = IO(new ThreshAndShiftIO(genThresh))
  val loadKernel = IO(Input(Valid(UInt(log2Up(kernel.numKernels).W))))
  val kernelNum = RegInit(0.U(log2Up(kernel.numKernels).W))

  when(loadKernel.valid) {
    kernelNum := loadKernel.bits
  }
  val threshWithIndex = thresh.values.zipWithIndex
  val shiftWithIndex = kernel.dtype.shift.zipWithIndex
  tasIO.thresh := MuxLookup(
    kernelNum,
    0.S.asTypeOf(genThresh),
    threshWithIndex.map(x => (x._2.toInt.U -> x._1.toInt.S.asTypeOf(genThresh)))
  )
  tasIO.shift := MuxLookup(kernelNum, 0.U, shiftWithIndex.map(x => (x._2.toInt.U -> x._1.abs.U)))
  tasIO.shiftLeft := MuxLookup(kernelNum, true.B, shiftWithIndex.map(x => (x._2.toInt.U -> (x._1 == x._1.abs).B)))
}
