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
import dsptools._
import fft._
import interfaces.amba.axis._

class LMFEWrapper(layer: LMFEConfig, options: LayerOptions) extends Module with LBIRStream {
  val logger = LoggerFactory.getLogger("LMFEWrapper")
  // TODO: remove this
  val fftParams = FFTParams.fixed(
    dataWidth = 24,
    binPoint = 12,
    trimEnable = false,
    numPoints = layer.fftSize,
    decimType = DITDecimType,
    trimType = RoundHalfToEven,
    twiddleWidth = 16,
    useBitReverse = true,
    windowFunc = WindowFunctionTypes.None(), // We do windowing in this module, because of issues with this
    overflowReg = true,
    numAddPipes = 1,
    numMulPipes = 1,
    sdfRadix = "2",
    runTime = false,
    expandLogic = Array.fill(log2Up(layer.fftSize))(1),
    keepMSBorLSB = Array.fill(log2Up(layer.fftSize))(true)
  )

  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))
  val melEngine = Module(new MelEngine(fftParams, 20, 32))
  require(options.busWidthOut % 8 == 0) // TODO: hardcoded that melEngine gives 8 bit output
  val numBeats = options.busWidthOut / 8
  val (beatCounter, beatCounterWrap) = Counter(0 to numBeats, melEngine.io.outStream.fire, outStream.fire)
  val outputBuffer = RegInit(VecInit(Seq.fill(numBeats)(0.U(8.W))))

  inStream.ready := melEngine.io.fftIn.ready
  melEngine.io.fftIn.valid := inStream.valid
  melEngine.io.fftIn.bits.real := inStream.bits.asTypeOf(melEngine.io.fftIn.bits.real)
  melEngine.io.fftIn.bits.imag := 0.U.asTypeOf(melEngine.io.fftIn.bits.imag)
  melEngine.io.lastFft := inStream.last

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
