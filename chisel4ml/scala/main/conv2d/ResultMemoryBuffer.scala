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
import interfaces.amba.axis.AXIStream
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import chisel4ml.HasLBIRStreamParameters


class ResultMemoryBuffer[O <: Bits](implicit val p: Parameters) extends Module 
with HasSequentialConvParameters
with HasLBIRStreamParameters {
  val io = IO(new Bundle {
    val outStream = AXIStream(UInt(outWidth.W))
    val result = Flipped(Decoupled(cfg.output.getType[O]))
  })
  val numRegs = if (cfg.output.numParams >= numBeatsOut) numBeatsOut else cfg.output.numParams
  val regs = Reg(Vec(numRegs, UInt(cfg.output.dtype.bitwidth.W)))
  val (totalCounter, totalCounterWrap) = Counter(0 until cfg.output.numParams, io.result.fire)
  val (registerCounter, registerCounterWrap) = Counter(0 until numRegs, io.result.fire, totalCounterWrap)
  dontTouch(totalCounter)

  when(io.result.fire) {
    regs(registerCounter) := io.result.bits.asUInt
  }

  io.outStream.bits := regs.asUInt
  io.outStream.valid := RegNext(registerCounterWrap) || RegNext(totalCounterWrap)
  io.outStream.last := RegNext(totalCounterWrap)
  dontTouch(io.outStream.last)

  io.result.ready := io.outStream.ready
}
