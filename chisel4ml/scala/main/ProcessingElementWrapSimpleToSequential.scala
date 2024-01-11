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
import org.slf4j.LoggerFactory
import chisel3._
import chisel3.util._
import interfaces.amba.axis._
import org.chipsalliance.cde.config.{Field, Parameters}

case object DenseConfigField extends Field[DenseConfig]

trait HasDenseParameters extends HasLBIRStreamParameters {
  type T = DenseConfig
  val p: Parameters
  val cfg = p(DenseConfigField)
  val inWidth = numBeatsIn * cfg.input.dtype.bitwidth
  val outWidth = numBeatsOut * cfg.output.dtype.bitwidth
}

class ProcessingElementWrapSimpleToSequential(implicit val p: Parameters)
    extends Module
    with HasLBIRStream[Vec[UInt]]
    with HasLBIRStreamParameters
    with HasDenseParameters {
  val logger = LoggerFactory.getLogger(this.getClass())

  val inStream = IO(Flipped(AXIStream(Vec(numBeatsIn, UInt(cfg.input.dtype.bitwidth.W)))))
  val outStream = IO(AXIStream(Vec(numBeatsOut, UInt(cfg.output.dtype.bitwidth.W))))
  val inputBuffer = RegInit(
    VecInit(Seq.fill(cfg.input.numTransactions(inWidth))(0.U(inWidth.W)))
  )
  val outputBuffer = RegInit(
    VecInit(Seq.fill(cfg.output.numTransactions(outWidth))(0.U(outWidth.W)))
  )

  logger.info(s"""Created new ProcessingElementWrapSimpleToSequential module. Number of input transactions:
                 |${cfg.input.numTransactions(inWidth)}, number of output transactions is:
                 |${cfg.output.numTransactions(outWidth)}, busWidthIn: ${inWidth},
                 | busWidthOut: ${outWidth}.""".stripMargin.replaceAll("\n", ""))

  val (inputCntValue, inputCntWrap) = Counter(inStream.fire, cfg.input.numTransactions(inWidth))
  val (outputCntValue, outputCntWrap) = Counter(outStream.fire, cfg.output.numTransactions(outWidth))
  val outputBufferFull = RegInit(false.B)

  // (combinational) computational module
  val peSimple = Module(ProcessingElementSimple(cfg))

  /** *** INPUT DATA INTERFACE ****
    */
  inStream.ready := !outputBufferFull
  when(inStream.fire) {
    inputBuffer(inputCntValue) := inStream.bits.asUInt
  }

  /** *** CONNECT INPUT AND OUTPUT REGSITERS WITH THE PE ****
    */
  peSimple.in := inputBuffer.asTypeOf(peSimple.in)
  when(RegNext(inStream.last)) {
    outputBuffer := peSimple.out.asTypeOf(outputBuffer)
    outputBufferFull := true.B
  }.elsewhen(outStream.last) {
    outputBufferFull := false.B
  }

  /** *** OUTPUT DATA INTERFACE ****
    */
  outStream.valid := outputBufferFull
  outStream.bits := outputBuffer(outputCntValue).asTypeOf(outStream.bits)
  outStream.last := outputCntWrap
}
