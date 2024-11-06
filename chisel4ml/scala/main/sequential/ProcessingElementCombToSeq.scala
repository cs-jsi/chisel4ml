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
import chisel4ml.util.risingEdge
import interfaces.amba.axis._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import org.chipsalliance.cde.config.Parameters
import services.Accelerator

object ProcessingElementCombToSeq {
  def apply(accel: Accelerator)(implicit p: Parameters) = {
    val idt = accel.layers.head.get.input.dtype
    val odt = accel.layers.last.get.output.dtype
    (idt.quantization, idt.signed, odt.quantization, odt.signed) match {
      case (BINARY, _, BINARY, _)           => new ProcessingElementCombToSeq[Bool, Bool]
      case (BINARY, _, UNIFORM, false)      => new ProcessingElementCombToSeq[Bool, UInt]
      case (BINARY, _, UNIFORM, true)       => new ProcessingElementCombToSeq[Bool, SInt]
      case (UNIFORM, false, BINARY, _)      => new ProcessingElementCombToSeq[UInt, Bool]
      case (UNIFORM, true, BINARY, _)       => new ProcessingElementCombToSeq[SInt, Bool]
      case (UNIFORM, true, UNIFORM, true)   => new ProcessingElementCombToSeq[SInt, SInt]
      case (UNIFORM, true, UNIFORM, false)  => new ProcessingElementCombToSeq[SInt, UInt]
      case (UNIFORM, false, UNIFORM, true)  => new ProcessingElementCombToSeq[UInt, SInt]
      case (UNIFORM, false, UNIFORM, false) => new ProcessingElementCombToSeq[UInt, UInt]
      case _                                => throw new RuntimeException
    }
  }
}

class ProcessingElementCombToSeq[I <: Bits, O <: Bits](implicit val p: Parameters)
    extends Module
    with HasAXIStream
    with HasAXIStreamParameters
    with HasParameterLogging
    with HasLayerWrapSeq {
  logParameters
  val CombModule = Module(new ProcessingPipelineCombinational(_cfg))
  val inStream = IO(Flipped(AXIStream(chiselTypeOf(CombModule.in.head), numBeatsIn)))
  val outStream = IO(AXIStream(chiselTypeOf(CombModule.out.head), numBeatsOut))
  val inputBuffer = RegInit(
    VecInit.fill(_cfg.head.input.numTransactions(numBeatsIn), numBeatsIn)(RegInit(_cfg.head.input.zero[I]))
  )
  dontTouch(inputBuffer)
  require(inputBuffer.flatten.length >= inStream.beats)
  val outputBuffer = RegInit(
    VecInit.fill(_cfg.last.output.numTransactions(numBeatsOut), numBeatsOut)(RegInit(_cfg.last.output.zero[O]))
  )
  dontTouch(outputBuffer)
  require(outputBuffer.flatten.length >= outStream.beats)

  val (inputCntValue, _) = Counter(inStream.fire, _cfg.head.input.numTransactions(numBeatsIn))
  val (outputCntValue, outputCntWrap) = Counter(outStream.fire, _cfg.last.output.numTransactions(numBeatsOut))
  val outputBufferFull = RegInit(false.B)

  // INPUT DATA INTERFACE
  inStream.ready := !outputBufferFull
  when(inStream.fire) {
    inputBuffer(inputCntValue) := inStream.bits
  }

  // CONNECT INPUT AND OUTPUT REGSITERS WITH THE PE
  CombModule.in := inputBuffer.flatten.slice(0, CombModule.in.length)
  val cond = RegNext(risingEdge(inStream.last))
  dontTouch(cond)
  when(cond) {
    outputBuffer := CombModule.out.asTypeOf(outputBuffer)
    outputBufferFull := true.B
  }.elsewhen(outStream.last) {
    outputBufferFull := false.B
  }

  // OUTPUT DATA INTERFACE
  outStream.valid := outputBufferFull
  outStream.bits := outputBuffer(outputCntValue)
  outStream.last := outputCntWrap
}
