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
import chisel4ml.LBIRStream
import lbir.LMFEConfig
import org.slf4j.LoggerFactory
import services.LayerOptions
import melengine._
import interfaces.amba.axis._
import chisel4ml.implicits._
import chisel3.experimental.FixedPoint

class LMFEWrapper(layer: LMFEConfig, options: LayerOptions) extends Module with LBIRStream {
  val logger = LoggerFactory.getLogger("LMFEWrapper")

  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))
  val melEngine = Module(
    new MelEngine(
      layer.fftSize,
      layer.numMels,
      layer.numFrames,
      layer.melFilters,
      FixedPoint(options.busWidthIn.W, layer.input.dtype.shift(0).BP)
    )
  )
  val numBeats = options.busWidthOut / 8
  val (beatCounter, beatCounterWrap) = Counter(0 to numBeats, melEngine.io.outStream.fire, outStream.fire)
  val (transactionCounter, _) = Counter(0 to layer.input.numTransactions(options.busWidthIn))
  dontTouch(transactionCounter)
  val outputBuffer = RegInit(VecInit(Seq.fill(numBeats)(0.U(8.W))))

  inStream.ready := melEngine.io.inStream.ready
  melEngine.io.inStream.valid := inStream.valid
  melEngine.io.inStream.bits := inStream.bits.asTypeOf(melEngine.io.inStream.bits)
  melEngine.io.inStream.last := inStream.last

  when(melEngine.io.outStream.fire) {
    outputBuffer(beatCounter) := melEngine.io.outStream.bits.asUInt
  }

  val last = RegInit(false.B)
  when(melEngine.io.outStream.last) {
    last := true.B
  }.elsewhen(last && outStream.fire) {
    last := false.B
  }

  outStream.valid := beatCounter === numBeats.U
  melEngine.io.outStream.ready := beatCounter < numBeats.U
  outStream.bits := outputBuffer.asUInt
  outStream.last := last
}
