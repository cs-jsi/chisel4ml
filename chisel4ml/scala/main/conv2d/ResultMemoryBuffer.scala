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
import chisel4ml.implicits._
import chisel3._
import services.LayerOptions
import interfaces.amba.axis.AXIStream
import chisel3.util._

class ResultMemoryBuffer[O <: Bits](output: lbir.QTensor, options: LayerOptions) extends Module {
  val io = IO(new Bundle {
    val outStream = AXIStream(UInt(options.busWidthOut.W))
    val result = Flipped(Decoupled(output.getType[O]))
  })
  val numRegs = if (output.numParams >= output.paramsPerWord) output.paramsPerWord else output.numParams
  val regs = Reg(Vec(numRegs, UInt(output.dtype.bitwidth.W)))
  val (regsCntVal, regsCntWrap) = Counter(0 until numRegs, io.result.fire)
  val (_, totalCntWrap) = Counter(0 until output.numParams, io.result.fire)

  when(io.result.fire) {
    regs(regsCntVal) := io.result.bits.asUInt
  }

  io.outStream.bits := regs.asUInt
  io.outStream.valid := RegNext(regsCntWrap)
  io.outStream.last := RegNext(totalCntWrap)

  io.result.ready := io.outStream.ready
}
