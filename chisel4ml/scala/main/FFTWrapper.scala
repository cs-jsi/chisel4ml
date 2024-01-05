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
import lbir.FFTConfig
import org.slf4j.LoggerFactory
import services.LayerOptions
import dsptools._
import fft._
import interfaces.amba.axis._

class FFTWrapper(layer: FFTConfig, options: LayerOptions) extends Module with LBIRStream {
  val logger = LoggerFactory.getLogger("FFTWrapper")

  val fftParams = FFTParams.fixed(
    dataWidth = 24,
    binPoint = 12,
    trimEnable = false,
    numPoints = layer.fftSize,
    decimType = DITDecimType,
    trimType = RoundHalfToEven,
    twiddleWidth = 16,
    useBitReverse = true,
    overflowReg = true,
    numAddPipes = 1,
    numMulPipes = 1,
    sdfRadix = "2",
    runTime = false,
    expandLogic = Array.fill(log2Up(layer.fftSize))(1),
    keepMSBorLSB = Array.fill(log2Up(layer.fftSize))(true)
  )

  val window = VecInit(layer.winFn.map(_.F(16.BP)))

  require(
    options.busWidthOut == layer.output.dtype.bitwidth,
    s"This module requires buswidhts to equal the input/output datatypes. " +
      s"${options.busWidthOut} != ${layer.output.dtype.bitwidth}"
  )
  val inStream = IO(Flipped(AXIStream(UInt(options.busWidthIn.W))))
  val outStream = IO(AXIStream(UInt(options.busWidthOut.W)))

  val sdffft = Module(new SDFFFT(fftParams))

  // Fix discrepancy between last signal semantics of LBIRDriver and FFT.
  // FFT has per frame last signals, LBIRDriver per tensor last signal.
  val (fftCounter, fftCounterWrap) = Counter(0 until fftParams.numPoints, inStream.fire)
  val (_, outCounterWrap) = Counter(0 until layer.numFrames, sdffft.io.lastOut)

  object fftState extends ChiselEnum {
    val sWAIT = Value(0.U)
    val sREADY = Value(1.U)
  }
  val state = RegInit(fftState.sREADY)

  when(state === fftState.sREADY && fftCounterWrap) {
    state := fftState.sWAIT
  }.elsewhen(state === fftState.sWAIT && sdffft.io.lastOut) {
    state := fftState.sREADY
  }

  inStream.ready := state === fftState.sREADY
  sdffft.io.in.valid := inStream.valid && state === fftState.sREADY
  val currWindow = window(fftCounter).asUInt.zext
  dontTouch(currWindow)
  // U(12, 0) x S(0, 16) => S(12, 16) >> 4 => S(12,12)
  val windowedSignal = (inStream.bits.asSInt * currWindow) >> 4
  sdffft.io.in.bits.real := windowedSignal.asTypeOf(sdffft.io.in.bits.real)
  sdffft.io.in.bits.imag := 0.U.asTypeOf(sdffft.io.in.bits.imag)
  sdffft.io.lastIn := inStream.last || fftCounterWrap

  sdffft.io.out.ready := outStream.ready
  outStream.valid := sdffft.io.out.valid
  outStream.bits := sdffft.io.out.bits.real.asTypeOf(outStream.bits)
  outStream.last := outCounterWrap
}
