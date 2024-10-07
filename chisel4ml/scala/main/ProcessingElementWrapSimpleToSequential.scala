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

import chisel4ml.implicits._
import lbir.DenseConfig
import chisel3._
import chisel3.util._
import interfaces.amba.axis._
import org.chipsalliance.cde.config.{Field, Parameters}
import chisel4ml.logging.HasParameterLogging

case object DenseConfigField extends Field[DenseConfig]

trait HasDenseParameters extends HasLBIRStreamParameters[DenseConfig] {
  type T = DenseConfig
  val p: Parameters
  val cfg = p(DenseConfigField)
}

class ProcessingElementWrapSimpleToSequential[I <: Bits, O <: Bits](implicit val p: Parameters)
    extends Module
    with HasLBIRStream
    with HasLBIRStreamParameters[DenseConfig]
    with HasDenseParameters
    with HasParameterLogging {
  logParameters
  val inStream = IO(Flipped(AXIStream(cfg.input.getType[I], numBeatsIn)))
  val outStream = IO(AXIStream(cfg.output.getType[O], numBeatsOut))
  val inputBuffer = RegInit(
    VecInit(Seq.fill(cfg.input.numTransactions(inWidth))(0.U(inWidth.W).asInstanceOf[I]))
  )
  val outputBuffer = RegInit(
    VecInit(Seq.fill(cfg.output.numTransactions(outWidth))(0.U(outWidth.W).asInstanceOf[O]))
  )

  val (inputCntValue, inputCntWrap) = Counter(inStream.fire, cfg.input.numTransactions(inWidth))
  val (outputCntValue, outputCntWrap) = Counter(outStream.fire, cfg.output.numTransactions(outWidth))
  val outputBufferFull = RegInit(false.B)

  // (combinational) computational module
  val peSimple = Module(ProcessingElementSimple(cfg))

  /** *** INPUT DATA INTERFACE ****
    */
  inStream.ready := !outputBufferFull
  when(inStream.fire) {
    inputBuffer(inputCntValue) := inStream.bits
  }

  /** *** CONNECT INPUT AND OUTPUT REGSITERS WITH THE PE ****
    */
  peSimple.in := inputBuffer
  when(RegNext(inStream.last)) {
    outputBuffer := peSimple.out
    outputBufferFull := true.B
  }.elsewhen(outStream.last) {
    outputBufferFull := false.B
  }

  /** *** OUTPUT DATA INTERFACE ****
    */
  outStream.valid := outputBufferFull
  outStream.bits := outputBuffer(outputCntValue)
  outStream.last := outputCntWrap
}
