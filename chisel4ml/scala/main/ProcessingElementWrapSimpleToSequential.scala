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

import interfaces.amba.axis._
import _root_.chisel4ml.util.log2
import _root_.chisel4ml.LBIRStream
import _root_.chisel4ml.implicits._
import _root_.lbir.{DenseConfig}
import _root_.services.LayerOptions
import _root_.scala.math

import _root_.org.slf4j.Logger
import _root_.org.slf4j.LoggerFactory

class ProcessingElementWrapSimpleToSequential(layer: DenseConfig, options: LayerOptions) extends Module with LBIRStream {
    val logger = LoggerFactory.getLogger(this.getClass())

    val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
    val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))
    val inputBuffer  = RegInit(VecInit(Seq.fill(layer.input.numTransactions(options.busWidthIn))(0.U(options.busWidthIn.W))))
    val outputBuffer = RegInit(VecInit(Seq.fill(layer.output.numTransactions(options.busWidthOut))(0.U(options.busWidthOut.W))))

    logger.info(s"""Created new ProcessingElementWrapSimpleToSequential module. Number of input transactions:
                   |${layer.input.numTransactions(options.busWidthIn)}, number of output transactions is:
                   |${layer.output.numTransactions(options.busWidthOut)}, busWidthIn: ${options.busWidthIn},
                   | busWidthOut: ${options.busWidthOut}.""".stripMargin.replaceAll("\n", ""))


    val (inputCntValue, inputCntWrap) = Counter(inStream.fire, layer.input.numTransactions(options.busWidthIn))
    val (outputCntValue, outputCntWrap) = Counter(outStream.fire, layer.output.numTransactions(options.busWidthOut))
    val outputBufferFull = RegInit(false.B)

    // (combinational) computational module
    val peSimple = Module(ProcessingElementSimple(layer))


    /***** INPUT DATA INTERFACE *****/
    inStream.ready := !outputBufferFull
    when(inStream.fire) {
        inputBuffer(inputCntValue) := inStream.bits
    }

    /***** CONNECT INPUT AND OUTPUT REGSITERS WITH THE PE *****/
    peSimple.io.in := inputBuffer.asUInt
    when (RegNext(inStream.last)) {
        outputBuffer :=  peSimple.io.out.asTypeOf(outputBuffer)
        outputBufferFull := true.B
    } .elsewhen(outStream.last) {
        outputBufferFull := false.B
    }

    /***** OUTPUT DATA INTERFACE *****/
    outStream.valid := outputBufferFull
    outStream.bits  := outputBuffer(outputCntValue)
    outStream.last  := outputCntWrap
}
