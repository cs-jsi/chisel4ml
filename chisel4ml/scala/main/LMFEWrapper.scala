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
package chisel4ml

import chisel3._
import chisel3.util._
import lbir.LMFEConfig
import org.slf4j.LoggerFactory
import melengine._
import interfaces.amba.axis._
import chisel4ml.implicits._
import chisel3.experimental.FixedPoint
import org.chipsalliance.cde.config.{Parameters, Field}

case object LMFEConfigField extends Field[LMFEConfig]

trait HasLMFEParameters extends HasLBIRStreamParameters {
  type T = LMFEConfig
  val p: Parameters
  val cfg = p(LMFEConfigField)
  val inWidth = numBeatsIn * cfg.input.dtype.bitwidth
  val outWidth = numBeatsOut * cfg.output.dtype.bitwidth
  require(numBeatsIn == 1)
  require(numBeatsOut == 1)
}

class LMFEWrapper(implicit val p: Parameters) extends Module 
with HasLBIRStream
with HasLBIRStreamParameters
with HasLMFEParameters {
  val logger = LoggerFactory.getLogger("LMFEWrapper")

  val inStream = IO(Flipped(AXIStream(UInt(cfg.input.dtype.bitwidth.W))))
  val outStream = IO(AXIStream(UInt(8.W)))
  val melEngine = Module(
    new MelEngine(
      cfg.fftSize,
      cfg.numMels,
      cfg.numFrames,
      cfg.melFilters,
      FixedPoint(cfg.input.dtype.bitwidth.W, cfg.input.dtype.shift(0).BP)
    )
  )
  val (beatCounter, beatCounterWrap) = Counter(0 to numBeatsOut, melEngine.io.outStream.fire, outStream.fire)
  val (transactionCounter, _) = Counter(0 to cfg.input.numTransactions(inWidth))
  dontTouch(transactionCounter)
  val outputBuffer = RegInit(VecInit(Seq.fill(numBeatsOut)(0.U(8.W))))

  inStream.ready := melEngine.io.inStream.ready
  melEngine.io.inStream.valid := inStream.valid
  melEngine.io.inStream.bits := inStream.bits(0).asTypeOf(melEngine.io.inStream.bits)
  melEngine.io.inStream.last := inStream.last

  when(melEngine.io.outStream.fire) {
    outputBuffer(beatCounter) := melEngine.io.outStream.bits.asUInt
  }

  val last = RegInit(false.B)
  when(melEngine.io.outStream.last) {
    last := true.B
  }.elsewhen(last && outStream.fire) {
    last := false.B
  }

  outStream.valid := beatCounter === numBeatsOut.U
  melEngine.io.outStream.ready := beatCounter < numBeatsOut.U
  outStream.bits := outputBuffer.asUInt
  outStream.last := last
}
