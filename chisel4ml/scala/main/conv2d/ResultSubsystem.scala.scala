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
package chisel4ml.conv2d
import chisel3._
import chisel4ml.MemWordSize
import chisel4ml.implicits._
import services.LayerOptions
import memories.MemoryGenerator
import interfaces.amba.axis.AXIStream
import chisel3.util.Decoupled

class ResultSubsystem[O <: Bits](output: lbir.QTensor, options: LayerOptions, genOut: O) extends Module {
  val io = IO(new Bundle {
    val outStream = AXIStream(UInt(options.busWidthOut.W))
    val result = Decoupled(genOut.cloneType)
  })

  val resMem = Module(MemoryGenerator.SRAM(depth = output.memDepth, width = MemWordSize.bits)) // ni potrebno?
  val rmb = Module(new ResultMemoryBuffer[O](genOut = genOut, output = output))
}
