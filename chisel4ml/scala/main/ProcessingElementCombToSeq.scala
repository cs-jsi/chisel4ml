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
import org.chipsalliance.cde.config.Parameters

class ProcessingElementCombToSeq(implicit val p: Parameters)
    extends Module
    with HasLBIRStream
    with HasLBIRStreamParameters
    with HasParameterLogging
    with HasIOContext {
  logParameters
  val inStream = IO(Flipped(AXIStream(_cfg.head.input.getType[ioc.I], numBeatsIn)))
  val outStream = IO(AXIStream(_cfg.last.output.getType[ioc.O], numBeatsOut))
  val CombModule = Module(new ProcessingPipelineSimple(_cfg))
  val inputBuffer = RegInit(
    VecInit.fill(_cfg.head.input.numTransactions(numBeatsIn), numBeatsIn)(RegInit(_cfg.head.input.gen[ioc.I]))
  )
  dontTouch(inputBuffer)
  require(inputBuffer.flatten.length >= inStream.beats)
  val outputBuffer = RegInit(
    VecInit.fill(_cfg.last.output.numTransactions(numBeatsOut), numBeatsOut)(RegInit(_cfg.last.output.gen[ioc.O]))
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
