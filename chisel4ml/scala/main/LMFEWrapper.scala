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

import _root_.chisel3._
import _root_.chisel3.util._
import _root_.chisel4ml.LBIRStream
import _root_.lbir.LMFEConfig
import _root_.org.slf4j.LoggerFactory
import _root_.services.LayerOptions
import afe._
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

  inStream.ready := melEngine.io.fftIn.ready
  melEngine.io.fftIn.valid := inStream.valid
  melEngine.io.fftIn.bits.real := inStream.bits.asTypeOf(melEngine.io.fftIn.bits.real)
  melEngine.io.fftIn.bits.imag := 0.U.asTypeOf(melEngine.io.fftIn.bits.imag)
  melEngine.io.lastFft := inStream.last

  outStream.valid := melEngine.io.outStream.valid
  melEngine.io.outStream.ready := outStream.ready
  outStream.bits := melEngine.io.outStream.bits.asUInt
  outStream.last := melEngine.io.outStream.last
}
