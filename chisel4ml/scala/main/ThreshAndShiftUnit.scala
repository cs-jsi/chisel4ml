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
import _root_.chisel4ml.lbir._
import _root_.lbir.{Layer, QTensor}
import chisel3._
import chisel3.experimental.ChiselEnum
import chisel3.util._

/** ThreshAndShiftUnit
  *
  */
class ThreshAndShiftUnit[A <: Bits: ThreshProvider](numKernels: Int, genThresh: A, layer: lbir.Layer)
extends Module {

  val io = IO(new Bundle {
    // interface to the DynamicNeuron module
    val thresh    = Output(genThresh)
    val shift     = Output(UInt(8.W))
    val shiftLeft = Output(Bool())

    // control interface
    val start      = Input(Bool())
    val nextKernel = Input(Bool())
  })

  val kernelNum = RegInit(0.U(reqWidth(numKernels).W))

  when (io.start) {
    kernelNum := 0.U
  }.elsewhen(io.nextKernel) {
    kernelNum := kernelNum + 1.U
  }

  val threshWithIndex = layer.thresh.get.values.zipWithIndex
  val shiftWithIndex  = layer.thresh.get.dtype.get.shift.zipWithIndex
  io.thresh    := MuxLookup(kernelNum,
                            0.S.asTypeOf(genThresh),
                            threshWithIndex.map(x => (x._2.toInt.U -> x._1.toInt.S.asTypeOf(genThresh))))
  io.shift     := MuxLookup(kernelNum,
                            0.U,
                            shiftWithIndex.map(x => (x._2.toInt.U -> x._1.abs.U)))
  io.shiftLeft := MuxLookup(kernelNum,
                            true.B,
                            shiftWithIndex.map(x => (x._2.toInt.U -> (x._1 == x._1.abs).B)))
}
