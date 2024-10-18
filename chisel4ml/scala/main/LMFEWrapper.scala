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
import chisel4ml.implicits._
import chisel4ml.logging.HasParameterLogging
import fixedpoint._
import interfaces.amba.axis._
import lbir.LMFEConfig
import melengine._
import org.chipsalliance.cde.config.{Field, Parameters}

case object LMFEConfigField extends Field[LMFEConfig]

trait HasLMFEParameters extends HasLBIRStreamParameters {
  val p: Parameters
  val cfg = p(LMFEConfigField)
  require(numBeatsIn == 1)
}

class LMFEWrapper(implicit val p: Parameters)
    extends Module
    with HasLBIRStream
    with HasLBIRStreamParameters
    with HasLMFEParameters
    with HasParameterLogging {
  logParameters
  val inStream = IO(Flipped(AXIStream(SInt(cfg.input.dtype.bitwidth.W), numBeatsIn)))
  val outStream = IO(AXIStream(UInt(cfg.output.dtype.bitwidth.W), numBeatsOut))
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
  val (transactionCounter, _) = Counter(0 to cfg.input.numTransactions(numBeatsIn))
  dontTouch(transactionCounter)
  val outputBuffer = RegInit(VecInit(Seq.fill(numBeatsOut)(0.U(8.W))))

  inStream.ready := melEngine.io.inStream.ready
  melEngine.io.inStream.valid := inStream.valid
  melEngine.io.inStream.bits := inStream.bits.asTypeOf(melEngine.io.inStream.bits)
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
  outStream.bits := outputBuffer
  outStream.last := last
}
