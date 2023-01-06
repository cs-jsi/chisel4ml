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
package chisel4ml.sequential

import chisel3._
import chisel3.util._

import _root_.chisel4ml.implicits._
import _root_.chisel4ml.bus.AXIStream
import _root_.chisel4ml.memory.SRAM
import _root_.chisel4ml.util.reqWidth
import _root_.lbir.Layer
import _root_.services.GenerateCircuitParams.Options

import _root_.org.slf4j.LoggerFactory

abstract class ProcessingElementSequential(layer: Layer, options: Options) extends Module {
    val logger = LoggerFactory.getLogger(this.getClass())

    val inputStreamWidth  = 32
    val outputStreamWidth = 32
    val memWordWidth      = 32

    val inSizeBits: Int = layer.input.get.totalBitwidth
    val numInTrans: Int = reqWidth(inSizeBits.toFloat / inputStreamWidth.toFloat)

    val outSizeBits: Int = layer.output.get.totalBitwidth
    val numOutTrans: Int = reqWidth(outSizeBits.toFloat / outputStreamWidth.toFloat)

    val io = IO(new Bundle {
        val inStream  = Flipped(new AXIStream(inputStreamWidth))
        val outStream = new AXIStream(outputStreamWidth)
    })

    logger.info(s"""Created new ProcessingElementSequential with inSizeBits: $inSizeBits,
                   | numInTrans: $numInTrans, outSizeBits: $outSizeBits, numOutTrans: $numOutTrans,
                   | inputStreamWidth: $inputStreamWidth, outputStreamWidth: $outputStreamWidth.
                   |""".stripMargin.replaceAll("\n", ""))
}

object ProcessingElementSequential {
    def apply(layer: Layer, options: Options) = new ProcessingElementSequentialWrapCombinational(layer, options)
}
