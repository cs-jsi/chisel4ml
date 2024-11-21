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
import interfaces.amba.axis._
import lbir.Datatype.QuantizationType.{BINARY, UNIFORM}
import org.chipsalliance.cde.config.Parameters
import services.Accelerator

object ProcessingElementCombToSeq {
  def apply(
    accel: Accelerator
  )(
    implicit p: Parameters
  ) = {
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

  object BufferState extends ChiselEnum {
    val sFULL, sNOT_FULL = Value
  }
}

trait HasPipelineRegisters extends HasLayerWrapSeq {
  val p: Parameters
  val numPipes = _cfg.length
}

class ProcessingElementCombToSeq[I <: Bits, O <: Bits](
  implicit val p: Parameters)
    extends Module
    with HasAXIStream
    with HasAXIStreamParameters
    with HasParameterLogging
    with HasLayerWrapSeq
    with HasPipelineRegisters {
  logParameters
  val CombModule = Module(new ProcessingPipelineCombinational(_cfg))
  val inStream = IO(Flipped(AXIStream(chiselTypeOf(CombModule.in.head), numBeatsIn)))
  val outStream = IO(AXIStream(chiselTypeOf(CombModule.out.head), numBeatsOut))
  val inputBuffer = RegInit(
    VecInit.fill(_cfg.head.input.numTransactions(numBeatsIn), numBeatsIn)(RegInit(_cfg.head.input.zero[I]))
  )
  import ProcessingElementCombToSeq.BufferState
  val inputBufferState = RegInit(BufferState.sNOT_FULL)
  dontTouch(inputBuffer)
  dontTouch(inputBufferState)
  require(inputBuffer.flatten.length >= inStream.beats)
  val outputBuffer = RegInit(
    VecInit.fill(_cfg.last.output.numTransactions(numBeatsOut), numBeatsOut)(RegInit(_cfg.last.output.zero[O]))
  )
  val outputBufferState = RegInit(BufferState.sNOT_FULL)
  dontTouch(outputBuffer)
  dontTouch(outputBufferState)
  require(outputBuffer.flatten.length >= outStream.beats)

  val (inputCntValue, inputCntWrap) = Counter(
    0 until _cfg.head.input.numTransactions(numBeatsIn),
    enable = inStream.fire,
    reset = inStream.last
  )
  dontTouch(inputCntWrap)
  val (outputCntValue, outputCntWrap) = Counter(
    0 until _cfg.last.output.numTransactions(numBeatsOut),
    enable = outStream.fire
  )
  dontTouch(outputCntWrap)

  // INPUT DATA INTERFACE
  inStream.ready := inputBufferState === BufferState.sNOT_FULL
  when(inStream.fire) {
    inputBuffer(inputCntValue) := inStream.bits
  }

  // CONNECT INPUT AND OUTPUT REGSITERS WITH THE PE
  CombModule.in := ShiftRegister(
    VecInit(inputBuffer.flatten.slice(0, CombModule.in.length)),
    numPipes
  )

  val cond = ShiftRegister(
    inStream.last,
    numPipes + 1
  )
  when(cond) {
    outputBuffer := CombModule.out.asTypeOf(outputBuffer)
    assert(outputBufferState === BufferState.sNOT_FULL)
  }
  dontTouch(cond)

  // OUTPUT DATA INTERFACE
  outStream.valid := outputBufferState === BufferState.sFULL
  outStream.bits := outputBuffer(outputCntValue)
  outStream.last := outputCntWrap

  // FINITE STATE MACHINE
  when(inStream.last) {
    inputBufferState := BufferState.sFULL
  }.elsewhen(outStream.last) {
    // Worst case FSM here, what if downstream PE can't block?
    inputBufferState := BufferState.sNOT_FULL
  }
  when(cond) {
    outputBufferState := BufferState.sFULL
  }.elsewhen(outStream.last) {
    outputBufferState := BufferState.sNOT_FULL
  }
}
