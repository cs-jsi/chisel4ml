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

import _root_.chisel4ml.implicits._
import interfaces.amba.axis.AXIStream
import _root_.chisel4ml.LBIRStream
import _root_.chisel4ml.util.log2
import _root_.lbir.{Layer}
import _root_.services.LayerOptions

import _root_.org.slf4j.LoggerFactory


abstract class ProcessingElementSequential(layer: Layer, options: LayerOptions) extends Module with LBIRStream {
    val logger = LoggerFactory.getLogger(this.getClass())

    val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
    val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))
}
