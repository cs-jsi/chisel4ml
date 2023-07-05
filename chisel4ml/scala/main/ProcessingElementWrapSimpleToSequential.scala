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
import _root_.chisel4ml.util.LbirUtil.log2
import _root_.lbir.{Layer}
import _root_.services.LayerOptions
import _root_.scala.math


class ProcessingElementWrapSimpleToSequential(layer: Layer, options: LayerOptions)
extends ProcessingElementSequential(layer, options) {
    val inputBuffer  = RegInit(VecInit(Seq.fill(numInTrans)(0.U(inputStreamWidth.W))))
    val outputBuffer = RegInit(VecInit(Seq.fill(numOutTrans)(0.U(outputStreamWidth.W))))

    val inputTransaction = inStream.valid && inStream.ready
    val outputTransaction = outStream.valid && outStream.ready

    val (inputCntValue, inputCntWrap) = Counter(inputTransaction, numInTrans-1)
    val (outputCntValue, outputCntWrap) = Counter(outputTransaction, numOutTrans-1)
    val outputBufferFull = RegInit(false.B)

    // (combinational) computational module
    val peSimple = Module(ProcessingElementSimple(layer))


    /***** INPUT DATA INTERFACE *****/
    inStream.ready := !outputBufferFull
    when(inputTransaction) {
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
